use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use h3_rs::config::MemoryConfig;
use h3_rs::dit::{Dit, DitCond, KeyframeCond};
use std::fs::File;
use std::io::Read;
use std::time::Instant;

fn read_h3lt_tensor<R: Read>(r: &mut R, dev: &Device) -> Result<Tensor> {
    let mut hdr = [0u8; 80];
    r.read_exact(&mut hdr)?;
    let dims = u32::from_le_bytes(hdr[4..8].try_into()?) as usize;
    let mut shape = Vec::with_capacity(dims);
    for i in 0..dims {
        let d = i64::from_le_bytes(hdr[8 + i * 8..16 + i * 8].try_into()?) as usize;
        shape.push(d);
    }
    let numel = u64::from_le_bytes(hdr[72..80].try_into()?) as usize;
    let mut data = vec![0f32; numel];
    let slice_u8 = unsafe {
        std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, numel * 4)
    };
    r.read_exact(slice_u8)?;
    shape.reverse();
    Ok(Tensor::from_vec(data, shape, dev)?)
}

fn load_cond(dev: &Device) -> Result<(DitCond, Vec<usize>, usize, [f32; 2])> {
    let mut f = File::open("/mnt/R2D/tmp/minimax_bridge/stage1_input_3s.edbin")?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    let mut header = [0i32; 4];
    for h in &mut header {
        let mut b = [0u8; 4];
        f.read_exact(&mut b)?;
        *h = i32::from_le_bytes(b);
    }
    let mut shifts = [0f32; 2];
    for s in &mut shifts {
        let mut b = [0u8; 4];
        f.read_exact(&mut b)?;
        *s = f32::from_le_bytes(b);
    }

    let context = read_h3lt_tensor(&mut f, dev)?;
    let token_tags_t = read_h3lt_tensor(&mut f, &Device::Cpu)?;
    let vid_lat = read_h3lt_tensor(&mut f, dev)?;
    let aud_lat = read_h3lt_tensor(&mut f, dev)?;
    let kf_lat = read_h3lt_tensor(&mut f, dev)?;
    let mut kf_idx_bytes = [0u8; 4];
    f.read_exact(&mut kf_idx_bytes)?;
    let kf_idx = i32::from_le_bytes(kf_idx_bytes) as usize;

    let token_tags_vec: Vec<u32> = token_tags_t.to_vec1::<f32>()?.iter().map(|&x| {
        if x > 0.0 && x < 1e-30 { x.to_bits() } else { x as u32 }
    }).collect();

    let cond = DitCond {
        text: context,
        tags: Some(token_tags_vec),
        keyframes: vec![KeyframeCond {
            frame_index: kf_idx,
            video: Some(kf_lat),
            audio: None,
        }],
        refs: vec![],
        seed: 42,
        visual_cond_noise_aug: 0.999,
        audio_cond_noise_aug: 1.0,
        denoise: None,
        control: None,
    };

    let video_shape = vid_lat.dims().to_vec();
    let audio_t = aud_lat.dim(3)?;
    Ok((cond, video_shape, audio_t, shifts))
}

fn main() -> Result<()> {
    println!("========================================================");
    println!("  Rank 4: OP-5 Streaming Double-Buffered DMA Parity Gate ");
    println!("========================================================");

    let dev = Device::new_cuda(0)?;
    let model_path = "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.safetensors";

    let (cond, video_shape, audio_t, _shifts) = load_cond(&dev)?;

    let mut mem = MemoryConfig::default();
    mem.dit_gpu_blocks = 30;
    mem.dit_vram_blocks = 30;
    mem.dit_ram_blocks = 17;
    mem.dit_disk_blocks = 0;
    mem.offload = h3_rs::weights::Placement::Disk;

    // Common inputs
    let sigmas = [1.0f64, 0.9826f64];
    let mut gen = h3_rs::rng::TorchGen::new(42);
    let nv = gen.randn_tensor(&video_shape, &dev)?;
    let na = gen.randn_tensor(&[1, 32, 2, audio_t], &dev)?;
    let xv = (&nv * sigmas[0])?;
    let xa = (&na * sigmas[0])?;

    // 1. Baseline: Serial Offload (H3_STREAMING=0)
    println!("\n--- 1. Baseline: Serial Offload (H3_STREAMING=0, 30 VRAM + 17 RAM) ---");
    std::env::set_var("H3_STREAMING", "0");
    let dit_base = Dit::load(model_path, &dev, &mem, None)?;
    assert!(dit_base.streaming.is_none(), "StreamingEngine must be disabled when H3_STREAMING=0");

    let run_base = dit_base.prepare(&cond, &video_shape, audio_t, vec![sigmas[0], sigmas[1]])?;
    dev.synchronize()?;

    let t0 = Instant::now();
    let (ov_base, _oa_base) = dit_base.forward(&run_base, &xv, &xa, sigmas[0])?;
    dev.synchronize()?;
    let t_base = t0.elapsed().as_secs_f64() * 1000.0;
    println!("   Serial Offload 1-Step Latency: {:.2} ms", t_base);

    drop(run_base);
    drop(dit_base);

    // 2. OP-5: Double-Buffered Staging Streaming (H3_STREAMING=1)
    println!("\n--- 2. OP-5: Double-Buffered Staging Streaming (H3_STREAMING=1, 30 VRAM + 17 RAM) ---");
    std::env::set_var("H3_STREAMING", "1");
    let dit_stream = Dit::load(model_path, &dev, &mem, None)?;
    assert!(dit_stream.streaming.is_some(), "StreamingEngine must be enabled when H3_STREAMING=1");

    let run_stream1 = dit_stream.prepare(&cond, &video_shape, audio_t, vec![sigmas[0], sigmas[1]])?;
    dev.synchronize()?;

    let t1 = Instant::now();
    let (ov_stream1, _oa_stream1) = dit_stream.forward(&run_stream1, &xv, &xa, sigmas[0])?;
    dev.synchronize()?;
    let t_stream1 = t1.elapsed().as_secs_f64() * 1000.0;
    println!("   Streaming Run 1 Latency: {:.2} ms", t_stream1);

    // 3. Test Path 2: Deterministic Repeat of Streaming Run (Check Race Conditions)
    println!("\n--- 3. Streaming Run 2: Deterministic Repeat (Race Condition Check) ---");
    let run_stream2 = dit_stream.prepare(&cond, &video_shape, audio_t, vec![sigmas[0], sigmas[1]])?;
    dev.synchronize()?;

    let t2 = Instant::now();
    let (ov_stream2, _oa_stream2) = dit_stream.forward(&run_stream2, &xv, &xa, sigmas[0])?;
    dev.synchronize()?;
    let t_stream2 = t2.elapsed().as_secs_f64() * 1000.0;
    println!("   Streaming Run 2 Latency: {:.2} ms", t_stream2);

    // 4. Parity and Determinism Verification
    println!("\n--- 4. Quality & Determinism Analysis ---");
    // Run1 vs Run2 (Race condition check)
    let diff_run = (&ov_stream1 - &ov_stream2)?.to_dtype(DType::F32)?;
    let max_run_diff = diff_run.abs()?.flatten_all()?.max(0)?.to_vec0::<f32>()?;
    println!("   Run1 vs Run2 Max Diff (Determinism): {:.6e}", max_run_diff);
    assert_eq!(max_run_diff, 0.0, "OP-5: Double-buffered streaming must be 100% deterministic (no stream race condition)!");

    // Streaming vs Serial Baseline (Mathematical Equivalence)
    let diff_base = (&ov_stream1 - &ov_base)?.to_dtype(DType::F32)?;
    let max_base_diff = diff_base.abs()?.flatten_all()?.max(0)?.to_vec0::<f32>()?;
    let norm_base = (ov_base.to_dtype(DType::F32)?.sqr()?.sum_all()?.to_vec0::<f32>()? as f64).sqrt();
    let norm_diff = (diff_base.sqr()?.sum_all()?.to_vec0::<f32>()? as f64).sqrt();
    let rel_l2 = norm_diff / norm_base;

    println!("   Streaming vs Serial Max Diff: {:.6e}", max_base_diff);
    println!("   Streaming vs Serial Relative L2: {:.6e}", rel_l2);
    assert!(rel_l2 < 0.001, "OP-5: Streaming must match serial baseline within float precision (<0.1%)!");

    let dma_saving = t_base - t_stream1;
    let speedup = t_base / t_stream1;
    println!("\n   PCIe Latency Hiding Savings: {:.2} ms per step ({:.2}x speedup)", dma_saving, speedup);

    println!("\n🎉 OP-5 Double-Buffered Streaming Parity Gate PASSED 100%!");
    Ok(())
}
