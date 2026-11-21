use std::sync::Arc;
use std::time::Instant;
use anyhow::{bail, Result};
use candle_core::cuda_backend::cudarc::driver::sys;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::cuda_backend::CudaStorageSlice;
use candle_core::{DType, Device, Storage, Tensor};
use h3_rs::streaming::{BlockPackLayout, ContiguousPinnedBuffer};
use h3_rs::weights::SafeFile;

fn check_cu(res: sys::CUresult, msg: &'static str) -> Result<()> {
    if res != sys::CUresult::CUDA_SUCCESS {
        bail!("CUDA Driver Error in {msg}: {res:?}");
    }
    Ok(())
}

fn tensor_storage_dptr(t: &Tensor) -> Result<u64> {
    let (st, l) = t.storage_and_layout();
    let Storage::Cuda(c) = &*st else {
        bail!("Expected CUDA tensor");
    };
    let stream = c.device.cuda_stream();
    let (ptr, _guard) = match &c.slice {
        CudaStorageSlice::I32(slice) => slice.device_ptr(&stream),
        CudaStorageSlice::U8(slice) => slice.device_ptr(&stream),
        CudaStorageSlice::BF16(slice) => slice.device_ptr(&stream),
        _ => bail!("Expected I32, U8, or BF16 storage slice"),
    };
    Ok(ptr + (l.start_offset() as u64 * t.dtype().size_in_bytes() as u64))
}

fn main() -> Result<()> {
    println!("====================================================================");
    println!("  NATIVE ACCELERATION TEST 1: Pinned DMA 2-Slot Overlap Benchmark   ");
    println!("  Target: MiniMax-H3 W4A4 DiT (48 Blocks, ~9.25 GB Weights Streaming) ");
    println!("====================================================================");

    let dev = Device::new_cuda(0)?;
    let Device::Cuda(cd) = &dev else { bail!("CUDA device required") };
    let stream_compute = cd.cuda_stream().cu_stream();

    // Check GPU device properties
    let model_path = if std::path::Path::new("/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_50b_w4a4_preswizzled.safetensors").exists() {
        "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_50b_w4a4_preswizzled.safetensors"
    } else {
        "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.safetensors"
    };
    println!("Loading block weight layout from: {}", model_path);
    let sf = SafeFile::open(model_path)?;
    let layout = BlockPackLayout::from_file(&sf, 0)?;
    let block_bytes = layout.total_bytes;
    let total_layers = 48usize;
    let total_weight_bytes = block_bytes * total_layers;
    println!("  Single Block W4A4 Weight Size: {:.2} MB ({} bytes)", block_bytes as f64 / 1e6, block_bytes);
    println!("  Full Model 48 Blocks Weights:   {:.2} GB ({} bytes)", total_weight_bytes as f64 / 1e9, total_weight_bytes);

    // Create a dedicated non-blocking CUDA stream for DMA copy
    let mut stream_copy: sys::CUstream = std::ptr::null_mut();
    check_cu(
        unsafe { sys::cuStreamCreate(&mut stream_copy, sys::CUstream_flags_enum::CU_STREAM_NON_BLOCKING as u32) },
        "cuStreamCreate(stream_copy)",
    )?;

    // ========================================================================
    // TEST 1.1: Baseline Pageable Host RAM -> VRAM (Synchronous Copy)
    // ========================================================================
    println!("\n--- TEST 1.1: Baseline Pageable Host RAM -> VRAM (Synchronous Copy) ---");
    let mut pageable_host_data = vec![0x55u8; block_bytes];
    let pageable_ptr = pageable_host_data.as_mut_ptr() as *const std::ffi::c_void;

    let vram_target_1 = Tensor::zeros(block_bytes.div_ceil(4), DType::I32, &dev)?;
    let vram_dptr_1 = tensor_storage_dptr(&vram_target_1)?;
    dev.synchronize()?;

    // Warmup
    unsafe {
        sys::cuMemcpyHtoD_v2(vram_dptr_1, pageable_ptr, block_bytes);
    }
    dev.synchronize()?;

    // Benchmark 10 iterations of 1 block to get accurate baseline
    let num_sample_blocks = 10;
    let t_base_start = Instant::now();
    for _ in 0..num_sample_blocks {
        unsafe {
            check_cu(
                sys::cuMemcpyHtoD_v2(vram_dptr_1, pageable_ptr, block_bytes),
                "cuMemcpyHtoD_v2(pageable)",
            )?;
        }
    }
    dev.synchronize()?;
    let t_base_elapsed = t_base_start.elapsed().as_secs_f64();
    let t_base_per_block_ms = (t_base_elapsed / num_sample_blocks as f64) * 1000.0;
    let bw_base_gbps = (block_bytes as f64 / 1e9) / (t_base_per_block_ms / 1000.0);
    let total_base_s = (t_base_per_block_ms * total_layers as f64) / 1000.0;

    println!("  Baseline Pageable Copy: {:.2} ms per block", t_base_per_block_ms);
    println!("  Baseline Bandwidth:     {:.2} GB/s", bw_base_gbps);
    println!("  Baseline 48 Blocks DMA: {:.2} s per step (EXPOSED BOTTLENECK)", total_base_s);

    // ========================================================================
    // TEST 1.2: 2-Slot Pinned Staging Ring Buffer (512MB: 2x 256MB Write-Combined)
    // ========================================================================
    println!("\n--- TEST 1.2: 2-Slot Pinned Staging Ring Buffer (cuMemHostAlloc Write-Combined) ---");
    let slot_alloc_bytes = block_bytes.max(256 * 1024 * 1024); // 256 MB per slot
    println!("  Allocating Slot 0 ({:.2} MB pinned RAM)...", slot_alloc_bytes as f64 / 1e6);
    let mut pinned_slot_0 = ContiguousPinnedBuffer::alloc(slot_alloc_bytes, true)?;
    println!("  Allocating Slot 1 ({:.2} MB pinned RAM)...", slot_alloc_bytes as f64 / 1e6);
    let mut pinned_slot_1 = ContiguousPinnedBuffer::alloc(slot_alloc_bytes, true)?;

    // Fill pinned buffers with test pattern
    unsafe {
        std::ptr::write_bytes(pinned_slot_0.as_mut_slice(block_bytes).as_mut_ptr(), 0xAA, block_bytes);
        std::ptr::write_bytes(pinned_slot_1.as_mut_slice(block_bytes).as_mut_ptr(), 0xBB, block_bytes);
    }

    // Allocate 2 static VRAM staging slots (Buffer A and Buffer B) on GPU
    let vram_target_0 = Tensor::zeros(block_bytes.div_ceil(4), DType::I32, &dev)?;
    let vram_dptr_0 = tensor_storage_dptr(&vram_target_0)?;
    let vram_dptr_1 = tensor_storage_dptr(&vram_target_1)?;
    dev.synchronize()?;

    // Measure raw pinned DMA bandwidth on stream_copy
    let t_pinned_start = Instant::now();
    for i in 0..num_sample_blocks {
        let (src_ptr, dst_dptr) = if (i & 1) == 0 {
            (pinned_slot_0.as_ptr() as *const std::ffi::c_void, vram_dptr_0)
        } else {
            (pinned_slot_1.as_ptr() as *const std::ffi::c_void, vram_dptr_1)
        };
        unsafe {
            check_cu(
                sys::cuMemcpyHtoDAsync_v2(dst_dptr, src_ptr, block_bytes, stream_copy),
                "cuMemcpyHtoDAsync_v2(pinned)",
            )?;
        }
    }
    unsafe {
        let _ = sys::cuStreamSynchronize(stream_copy);
    }
    let t_pinned_elapsed = t_pinned_start.elapsed().as_secs_f64();
    let t_pinned_per_block_ms = (t_pinned_elapsed / num_sample_blocks as f64) * 1000.0;
    let bw_pinned_gbps = (block_bytes as f64 / 1e9) / (t_pinned_per_block_ms / 1000.0);
    let total_pinned_s = (t_pinned_per_block_ms * total_layers as f64) / 1000.0;

    println!("  Pinned DMA Copy:        {:.2} ms per block", t_pinned_per_block_ms);
    println!("  Pinned DMA Bandwidth:   {:.2} GB/s (PCIe 3.0 x8/x16 line rate)", bw_pinned_gbps);
    println!("  Raw Pinned 48 Blocks:   {:.2} s per step (Physical lower bound)", total_pinned_s);
    println!("  -> Pure DMA Speedup:    {:.2}x over pageable!", bw_pinned_gbps / bw_base_gbps);

    // ========================================================================
    // TEST 1.3: Asynchronous Overlap Verification (Compute Stream vs DMA Copy Stream)
    // ========================================================================
    println!("\n--- TEST 1.3: Asynchronous Overlap Verification (Compute Stream vs Copy Stream) ---");
    // Create synchronization events
    let mut ev_copy_done_0: sys::CUevent = std::ptr::null_mut();
    let mut ev_copy_done_1: sys::CUevent = std::ptr::null_mut();
    let mut ev_compute_done_0: sys::CUevent = std::ptr::null_mut();
    let mut ev_compute_done_1: sys::CUevent = std::ptr::null_mut();

    unsafe {
        check_cu(sys::cuEventCreate(&mut ev_copy_done_0, sys::CUevent_flags_enum::CU_EVENT_DISABLE_TIMING as u32), "ev_copy_done_0")?;
        check_cu(sys::cuEventCreate(&mut ev_copy_done_1, sys::CUevent_flags_enum::CU_EVENT_DISABLE_TIMING as u32), "ev_copy_done_1")?;
        check_cu(sys::cuEventCreate(&mut ev_compute_done_0, sys::CUevent_flags_enum::CU_EVENT_DISABLE_TIMING as u32), "ev_compute_done_0")?;
        check_cu(sys::cuEventCreate(&mut ev_compute_done_1, sys::CUevent_flags_enum::CU_EVENT_DISABLE_TIMING as u32), "ev_compute_done_1")?;
        // Initialize compute_done events as recorded
        check_cu(sys::cuEventRecord(ev_compute_done_0, stream_compute), "init ev_compute_done_0")?;
        check_cu(sys::cuEventRecord(ev_compute_done_1, stream_compute), "init ev_compute_done_1")?;
    }

    // Prepare a synthetic compute workload on stream_compute matching 1 block time (e.g. GEMM matmuls)
    let m = 12104usize;
    let k = 3584usize;
    let n = 3584usize;
    let a_mat = Tensor::randn(0.0f32, 1.0, (m, k), &dev)?.to_dtype(DType::BF16)?;
    let b_mat = Tensor::randn(0.0f32, 1.0, (k, n), &dev)?.to_dtype(DType::BF16)?;
    dev.synchronize()?;

    // Measure isolated compute time per block
    let t_comp_start = Instant::now();
    let num_sim_blocks = 12usize;
    for _ in 0..num_sim_blocks {
        let _c = a_mat.matmul(&b_mat)?;
    }
    dev.synchronize()?;
    let t_comp_total = t_comp_start.elapsed().as_secs_f64();
    let t_comp_per_block_ms = (t_comp_total / num_sim_blocks as f64) * 1000.0;
    println!("  Isolated Compute Workload: {:.2} ms per block", t_comp_per_block_ms);

    // Now test CONCURRENT OVERLAPPED execution of Compute and Pinned DMA
    dev.synchronize()?;
    let t_overlap_start = Instant::now();

    // Prime the first block (Block 0 into Slot 0)
    unsafe {
        check_cu(
            sys::cuMemcpyHtoDAsync_v2(vram_dptr_0, pinned_slot_0.as_ptr() as *const std::ffi::c_void, block_bytes, stream_copy),
            "cuMemcpyHtoDAsync_v2(prime 0)",
        )?;
        check_cu(sys::cuEventRecord(ev_copy_done_0, stream_copy), "record ev_copy_done_0")?;
    }

    for bi in 0..num_sim_blocks {
        let slot_curr = bi & 1;
        let slot_next = (bi + 1) & 1;
        let (dptr_next, src_next, ev_copy_done_curr, ev_compute_done_curr, ev_copy_done_next, ev_compute_done_next) = if slot_curr == 0 {
            (vram_dptr_1, pinned_slot_1.as_ptr(), ev_copy_done_0, ev_compute_done_0, ev_copy_done_1, ev_compute_done_1)
        } else {
            (vram_dptr_0, pinned_slot_0.as_ptr(), ev_copy_done_1, ev_compute_done_1, ev_copy_done_0, ev_compute_done_0)
        };

        // 1. stream_compute waits for current block copy to finish
        unsafe {
            check_cu(sys::cuStreamWaitEvent(stream_compute, ev_copy_done_curr, 0), "wait ev_copy_done")?;
        }

        // 2. stream_copy prefetches NEXT block while stream_compute computes CURRENT block
        if bi + 1 < num_sim_blocks {
            unsafe {
                // stream_copy waits until stream_compute is done with slot_next
                check_cu(sys::cuStreamWaitEvent(stream_copy, ev_compute_done_next, 0), "wait ev_compute_done_next")?;
                // enqueue async DMA for next block
                check_cu(
                    sys::cuMemcpyHtoDAsync_v2(dptr_next, src_next as *const std::ffi::c_void, block_bytes, stream_copy),
                    "cuMemcpyHtoDAsync_v2(overlap next)",
                )?;
                check_cu(sys::cuEventRecord(ev_copy_done_next, stream_copy), "record ev_copy_done_next")?;
            }
        }

        // 3. stream_compute executes block compute
        let _c = a_mat.matmul(&b_mat)?;

        // 4. stream_compute records compute done on slot_curr
        unsafe {
            check_cu(sys::cuEventRecord(ev_compute_done_curr, stream_compute), "record ev_compute_done_curr")?;
        }
    }

    dev.synchronize()?;
    unsafe {
        let _ = sys::cuStreamSynchronize(stream_copy);
    }
    let t_overlap_total = t_overlap_start.elapsed().as_secs_f64();
    let t_serial_total = (t_comp_per_block_ms + t_pinned_per_block_ms) * num_sim_blocks as f64 / 1000.0;
    let t_serial_base_total = (t_comp_per_block_ms + t_base_per_block_ms) * num_sim_blocks as f64 / 1000.0;

    let exposed_dma_per_block_ms = ((t_overlap_total - t_comp_total) / num_sim_blocks as f64) * 1000.0;
    let exposed_dma_total_48_s = (exposed_dma_per_block_ms.max(0.0) * total_layers as f64) / 1000.0;

    println!("  Overlapped Wall-Clock Time:  {:.2} s (for {} simulated blocks)", t_overlap_total, num_sim_blocks);
    println!("  Serialized (Compute + Pinned): {:.2} s", t_serial_total);
    println!("  Serialized (Compute + Baseline): {:.2} s", t_serial_base_total);
    println!("  Exposed DMA Time per block:  {:.2} ms (vs {:.2} ms baseline)", exposed_dma_per_block_ms.max(0.0), t_base_per_block_ms);
    println!("  Exposed DMA across 48 blocks: {:.2} s (vs {:.2} s baseline)", exposed_dma_total_48_s, total_base_s);

    let dma_hiding_ratio = (1.0 - (exposed_dma_per_block_ms.max(0.0) / t_base_per_block_ms)) * 100.0;
    println!("  -> DMA Latency Hiding Efficiency: {:.1}%", dma_hiding_ratio);

    // ========================================================================
    // TEST 1.4: Bitwise Parity Gate (100% Bit-Exact check)
    // ========================================================================
    println!("\n--- TEST 1.4: Bitwise Parity Gate (100% Bit-Exact Parity) ---");
    let test_pattern = vec![0x12345678i32; block_bytes / 4];
    let mut pinned_verify = ContiguousPinnedBuffer::alloc(block_bytes, true)?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            test_pattern.as_ptr() as *const u8,
            pinned_verify.as_mut_slice(block_bytes).as_mut_ptr(),
            block_bytes,
        );
    }
    let gpu_verify = Tensor::zeros(block_bytes / 4, DType::I32, &dev)?;
    let gpu_verify_dptr = tensor_storage_dptr(&gpu_verify)?;
    unsafe {
        check_cu(
            sys::cuMemcpyHtoDAsync_v2(gpu_verify_dptr, pinned_verify.as_ptr() as *const std::ffi::c_void, block_bytes, stream_copy),
            "verify cuMemcpyHtoDAsync_v2",
        )?;
        let _ = sys::cuStreamSynchronize(stream_copy);
    }
    let back_to_cpu = gpu_verify.to_device(&candle_core::Device::Cpu)?;
    let back_vec = back_to_cpu.to_vec1::<i32>()?;
    let mut mismatch_count = 0usize;
    for (i, (&orig, &back)) in test_pattern.iter().zip(back_vec.iter()).enumerate() {
        if orig != back {
            mismatch_count += 1;
            if mismatch_count <= 5 {
                eprintln!("  Mismatch at word {i}: expected 0x{orig:X}, got 0x{back:X}");
            }
        }
    }
    println!("  Total Mismatched Words: {} / {}", mismatch_count, test_pattern.len());
    assert_eq!(mismatch_count, 0, "DMA transfer must be 100% bitwise exact!");
    println!("  -> Parity Verification: 100% BIT-EXACT PASSED (Cosine = 1.0000000)!");

    println!("\n====================================================================");
    println!("  SUMMARY: TEST 1 (Pinned DMA Overlap) RESULTS                       ");
    println!("  1. Baseline Pageable DMA:      {:.2} s / step ({:.2} GB/s)", total_base_s, bw_base_gbps);
    println!("  2. Pinned Staging DMA:         {:.2} s / step ({:.2} GB/s, {:.2}x bandwidth)", total_pinned_s, bw_pinned_gbps, bw_pinned_gbps / bw_base_gbps);
    println!("  3. Overlapped Exposed DMA:     {:.2} s / step (Saved: {:.2} s / step!)", exposed_dma_total_48_s, total_base_s - exposed_dma_total_48_s);
    println!("  4. Bitwise Parity:             100% Bit-Exact (0.0000% error)");
    println!("====================================================================");

    // Cleanup CUDA events and stream
    unsafe {
        let _ = sys::cuEventDestroy_v2(ev_copy_done_0);
        let _ = sys::cuEventDestroy_v2(ev_copy_done_1);
        let _ = sys::cuEventDestroy_v2(ev_compute_done_0);
        let _ = sys::cuEventDestroy_v2(ev_compute_done_1);
        let _ = sys::cuStreamDestroy_v2(stream_copy);
    }
    Ok(())
}
