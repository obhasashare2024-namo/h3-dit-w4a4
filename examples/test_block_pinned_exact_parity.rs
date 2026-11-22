use anyhow::Result;
use candle_core::{DType, Device, Tensor, Storage};
use candle_core::cuda_backend::cudarc::driver::{sys, DevicePtr};
use h3_rs::pinned_att::get_or_create_pinned_att;

fn main() -> Result<()> {
    println!("========================================================");
    println!("  DEF-108: Full 30-Chunk Bitwise Parity Gate            ");
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

    // 2. Setup problem dimensions (1088P 8s exact specifications)
    let s = 121_041usize;
    let groups = 28usize;
    let w = 64usize;
    let chunk = 4096usize;

    println!("Allocating PinnedAttBuffer: S={}, G={}, W={} (Total {:.2} MB)",
             s, groups, w, (s * groups * w * 2) as f64 / 1e6);
    let pbuf = get_or_create_pinned_att(s, groups, w)?;

    // Fill pinned buffer with deterministic pseudo-random data
    println!("Filling PinnedAttBuffer with deterministic pattern...");
    for g in 0..groups {
        let host_src = pbuf.host_ptr(g);
        let bytes_per_group = s * w * 2;
        let mut host_data = vec![0u8; bytes_per_group];
        for i in 0..(s * w) {
            // Write distinct deterministic bf16 values
            let val = ((g * 1337 + i * 31) % 65535) as u16;
            host_data[i * 2] = (val & 0xff) as u8;
            host_data[i * 2 + 1] = ((val >> 8) & 0xff) as u8;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(host_data.as_ptr(), host_src, bytes_per_group);
        }
    }

    // 3. Run Reference Serial Fetch on all chunks
    println!("\n--- Step 1: Running Reference Serial Fetch ---");
    let serial_staging = Tensor::zeros((groups, chunk, w), DType::BF16, &dev)?;
    let serial_gathered = Tensor::zeros((chunk, groups * w), DType::BF16, &dev)?;
    let mut serial_results = Vec::new();

    let mut st = 0;
    let mut chunk_idx = 0;
    while st < s {
        let n = chunk.min(s - st);
        let (stg, gth) = if n == chunk {
            (serial_staging.clone(), serial_gathered.clone())
        } else {
            (
                Tensor::zeros((groups, n, w), DType::BF16, &dev)?,
                Tensor::zeros((n, groups * w), DType::BF16, &dev)?,
            )
        };
        pbuf.fetch_chunk_gathered(st, n, &stg, &gth)?;
        dev.synchronize()?;
        serial_results.push(gth.to_device(&Device::Cpu)?);
        st += n;
        chunk_idx += 1;
    }
    println!("Serial fetch produced {} chunks.", serial_results.len());

    // 4. Run Stream Overlap Fetch with Tail-Aware Dynamic Staging
    println!("\n--- Step 2: Running Tail-Aware Stream Overlap Fetch ---");
    let staging = [
        Tensor::zeros((groups, chunk, w), DType::BF16, &dev)?,
        Tensor::zeros((groups, chunk, w), DType::BF16, &dev)?,
    ];
    let gathered = [
        Tensor::zeros((chunk, groups * w), DType::BF16, &dev)?,
        Tensor::zeros((chunk, groups * w), DType::BF16, &dev)?,
    ];
    let (tail_staging, tail_gathered) = if s % chunk != 0 {
        let rem = s % chunk;
        (
            Some(Tensor::zeros((groups, rem, w), DType::BF16, &dev)?),
            Some(Tensor::zeros((rem, groups * w), DType::BF16, &dev)?),
        )
    } else {
        (None, None)
    };

    let mut ev_copy: [sys::CUevent; 2] = [std::ptr::null_mut(), std::ptr::null_mut()];
    let mut ev_compute: [sys::CUevent; 2] = [std::ptr::null_mut(), std::ptr::null_mut()];
    unsafe {
        sys::cuEventCreate(&mut ev_copy[0], sys::CUevent_flags_enum::CU_EVENT_DISABLE_TIMING as u32);
        sys::cuEventCreate(&mut ev_copy[1], sys::CUevent_flags_enum::CU_EVENT_DISABLE_TIMING as u32);
        sys::cuEventCreate(&mut ev_compute[0], sys::CUevent_flags_enum::CU_EVENT_DISABLE_TIMING as u32);
        sys::cuEventCreate(&mut ev_compute[1], sys::CUevent_flags_enum::CU_EVENT_DISABLE_TIMING as u32);
    }

    let n0 = chunk.min(s);
    let stg0 = if n0 == chunk { &staging[0] } else { tail_staging.as_ref().unwrap() };
    pbuf.fetch_chunk_staging_stream(0, n0, stg0, stream_copy)?;
    unsafe {
        sys::cuEventRecord(ev_copy[0], stream_copy);
    }

    let mut stream_results = Vec::new();
    let mut st = 0;
    let mut cur = 0;
    while st < s {
        let n = chunk.min(s - st);
        let nxt_st = st + n;
        let nxt = 1 - cur;

        if nxt_st < s {
            let nxt_n = chunk.min(s - nxt_st);
            let stg_nxt = if nxt_n == chunk { &staging[nxt] } else { tail_staging.as_ref().unwrap() };
            unsafe {
                if st > 0 {
                    sys::cuStreamWaitEvent(stream_copy, ev_compute[nxt], 0);
                }
            }
            pbuf.fetch_chunk_staging_stream(nxt_st, nxt_n, stg_nxt, stream_copy)?;
            unsafe {
                sys::cuEventRecord(ev_copy[nxt], stream_copy);
            }
        }

        unsafe {
            sys::cuStreamWaitEvent(stream_compute, ev_copy[cur], 0);
        }

        let (stg_cur, gth_cur) = if n == chunk {
            (&staging[cur], &gathered[cur])
        } else {
            (tail_staging.as_ref().unwrap(), tail_gathered.as_ref().unwrap())
        };

        h3_rs::kernels::gather_groups_to_rows_inplace(gth_cur, stg_cur, groups, w)?;
        dev.synchronize()?;
        stream_results.push(gth_cur.to_device(&Device::Cpu)?);

        unsafe {
            sys::cuEventRecord(ev_compute[cur], stream_compute);
        }

        st = nxt_st;
        cur = nxt;
    }

    unsafe {
        sys::cuStreamSynchronize(stream_compute);
        sys::cuEventDestroy_v2(ev_copy[0]);
        sys::cuEventDestroy_v2(ev_copy[1]);
        sys::cuEventDestroy_v2(ev_compute[0]);
        sys::cuEventDestroy_v2(ev_compute[1]);
        sys::cuStreamDestroy_v2(stream_copy);
    }

    // 5. Verification Gate: Compare all chunks
    println!("\n--- Step 3: Verifying Bit-Exact Parity Across All 30 Chunks ---");
    assert_eq!(serial_results.len(), stream_results.len());
    let mut max_all_diff = 0.0f32;
    for (i, (s_res, m_res)) in serial_results.iter().zip(stream_results.iter()).enumerate() {
        let diff = (s_res - m_res)?.abs()?.max(0)?.max(0)?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        let n_tokens = s_res.dim(0)?;
        if diff > 0.0 || i == 0 || i == 28 || i == 29 {
            println!("  Chunk {:2} (n={:4}): Max Abs Diff = {:.6e}", i, n_tokens, diff);
        }
        if diff > max_all_diff {
            max_all_diff = diff;
        }
        assert_eq!(diff, 0.0, "Chunk {} failed parity test!", i);
    }

    println!("\nOverall Max Abs Diff across all 30 chunks: {:.6e}", max_all_diff);
    assert_eq!(max_all_diff, 0.0, "Must be 100% Bit-Exact across all 30 chunks!");

    println!("\n========================================================");
    println!("  ✅ DEF-108: ALL 30 CHUNKS 100% BIT-EXACT PASSED!      ");
    println!("========================================================");
    Ok(())
}
