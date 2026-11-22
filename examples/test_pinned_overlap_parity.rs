use anyhow::Result;
use candle_core::{DType, Device, Tensor, Storage};
use candle_core::cuda_backend::cudarc::driver::{sys, DevicePtr};
use h3_rs::pinned_att::get_or_create_pinned_att;
use std::time::Instant;

fn main() -> Result<()> {
    println!("========================================================");
    println!("  DEF-108: Pinned Overlap Stream & Parity Test          ");
    println!("========================================================");

    let dev = Device::new_cuda(0)?;
    let Device::Cuda(cd) = &dev else { anyhow::bail!("CUDA device required") };
    let stream_guard = cd.cuda_stream();
    let stream_compute = stream_guard.cu_stream();

    // 1. Create a dedicated non-blocking stream for copy operations
    let mut stream_copy: sys::CUstream = std::ptr::null_mut();
    unsafe {
        let res = sys::cuStreamCreate(&mut stream_copy, sys::CUstream_flags_enum::CU_STREAM_NON_BLOCKING as u32);
        assert_eq!(res, sys::CUresult::CUDA_SUCCESS, "cuStreamCreate failed");
    }

    // 2. Setup problem dimensions (1088P 8s specifications)
    let s = 121_041usize;
    let groups = 14usize;
    let w = 512usize; // 14 groups * 512 = 7168 inner dim
    let chunk = 4096usize;

    println!("Allocating PinnedAttBuffer: S={}, G={}, W={} (Total {:.2} GB)",
             s, groups, w, (s * groups * w * 2) as f64 / 1e9);
    let pbuf = get_or_create_pinned_att(s, groups, w)?;

    // 3. Test DtoH write_group_async with event sync
    println!("\n--- Test 1: DtoH Group Write with Event Overlap ---");
    let mut ev_compute: sys::CUevent = std::ptr::null_mut();
    let mut ev_copy: sys::CUevent = std::ptr::null_mut();
    unsafe {
        sys::cuEventCreate(&mut ev_compute, sys::CUevent_flags_enum::CU_EVENT_DISABLE_TIMING as u32);
        sys::cuEventCreate(&mut ev_copy, sys::CUevent_flags_enum::CU_EVENT_DISABLE_TIMING as u32);
    }

    // Generate random group tensor
    let o = Tensor::randn(0.0f32, 1.0f32, (s, w), &dev)?.to_dtype(DType::BF16)?;
    dev.synchronize()?;

    // Measure single stream DtoH
    let t0 = Instant::now();
    pbuf.write_group_async(0, &o)?;
    dev.synchronize()?;
    let t_single_dtoh = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Single-stream DtoH: {:.2} ms (124 MB)", t_single_dtoh);

    // Measure overlap DtoH on copy stream
    let (st_o, l_o) = o.storage_and_layout();
    let Storage::Cuda(c_o) = &*st_o else { anyhow::bail!("CUDA tensor required") };
    let (p_o, _guard) = match &c_o.slice {
        candle_core::cuda_backend::CudaStorageSlice::BF16(slice) => slice.device_ptr(&stream_guard),
        _ => anyhow::bail!("BF16 slice required"),
    };
    let dptr_o = p_o + (l_o.start_offset() * 2) as u64;
    let bytes_o = s * w * 2;

    let t1 = Instant::now();
    unsafe {
        // Record compute done on compute stream
        sys::cuEventRecord(ev_compute, stream_compute);
        // Copy stream waits for compute
        sys::cuStreamWaitEvent(stream_copy, ev_compute, 0);
        // Copy stream performs DtoH
        let dst_host = pbuf.host_ptr(0);
        let res = sys::cuMemcpyDtoHAsync_v2(dst_host as *mut std::ffi::c_void, dptr_o as sys::CUdeviceptr, bytes_o, stream_copy);
        assert_eq!(res, sys::CUresult::CUDA_SUCCESS);
        // Record copy done
        sys::cuEventRecord(ev_copy, stream_copy);
    }
    // Meanwhile compute stream does not wait, it can proceed!
    let t_overlap_launch = t1.elapsed().as_secs_f64() * 1000.0;
    println!("  Overlap DtoH launch time (compute stream free): {:.3} ms", t_overlap_launch);

    unsafe {
        sys::cuStreamSynchronize(stream_copy);
    }
    let t_overlap_total = t1.elapsed().as_secs_f64() * 1000.0;
    println!("  Overlap DtoH total complete: {:.2} ms", t_overlap_total);

    // 4. Test HtoD Gather Parity
    println!("\n--- Test 2: Double-Buffered Gather Parity ---");
    let staging = Tensor::zeros((groups, chunk, w), DType::BF16, &dev)?;
    let gathered_serial = Tensor::zeros((chunk, groups * w), DType::BF16, &dev)?;
    let gathered_stream = Tensor::zeros((chunk, groups * w), DType::BF16, &dev)?;

    // Serial fetch
    pbuf.fetch_chunk_gathered(0, chunk, &staging, &gathered_serial)?;
    dev.synchronize()?;

    // Stream fetch
    for g in 0..groups {
        let host_src = pbuf.host_ptr_offset(g, 0);
        let dst_byte_offset = g * chunk * w * 2;
        let bytes_per_panel = chunk * w * 2;
        let (st_s, l_s) = staging.storage_and_layout();
        let Storage::Cuda(c_s) = &*st_s else { anyhow::bail!("CUDA tensor required") };
        let (p_s, _guard) = match &c_s.slice {
            candle_core::cuda_backend::CudaStorageSlice::BF16(slice) => slice.device_ptr(&stream_guard),
            _ => anyhow::bail!("BF16 slice required"),
        };
        let dptr_s = p_s + (l_s.start_offset() * 2) as u64 + dst_byte_offset as u64;
        unsafe {
            let res = sys::cuMemcpyHtoDAsync_v2(dptr_s as sys::CUdeviceptr, host_src as *const std::ffi::c_void, bytes_per_panel, stream_copy);
            assert_eq!(res, sys::CUresult::CUDA_SUCCESS);
        }
    }
    unsafe {
        sys::cuEventRecord(ev_copy, stream_copy);
        sys::cuStreamWaitEvent(stream_compute, ev_copy, 0);
    }
    h3_rs::kernels::gather_groups_to_rows_inplace(&gathered_stream, &staging, groups, w)?;
    dev.synchronize()?;

    // Verify bitwise parity
    let diff = (gathered_serial - gathered_stream)?.abs()?.max(0)?.max(0)?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
    println!("  Max Abs Diff between Serial and Stream Gathered: {:.6e}", diff);
    assert_eq!(diff, 0.0, "Must be bitwise identical!");

    // Clean up
    unsafe {
        sys::cuEventDestroy_v2(ev_compute);
        sys::cuEventDestroy_v2(ev_copy);
        sys::cuStreamDestroy_v2(stream_copy);
    }

    println!("\n========================================================");
    println!("  ✅ DEF-108: Stream Overlap & Bitwise Parity PASSED!   ");
    println!("========================================================");
    Ok(())
}
