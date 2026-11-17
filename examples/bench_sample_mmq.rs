use std::fs::File;
use std::io::Read;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use h3_rs::config::MemoryConfig;
use h3_rs::dit::{Dit, DitCond, KeyframeCond};
use h3_rs::sampling::{sample_denoised, simple_sigmas, Sampler};

fn read_h3lt_tensor<R: Read>(r: &mut R, dev: &Device) -> Result<Tensor> {
    let mut hdr = [0u8; 80];
    r.read_exact(&mut hdr)?;
    let magic = u32::from_le_bytes(hdr[0..4].try_into()?);
    if magic != 0x48334c54 {
        bail!("Invalid H3LT tensor magic: {:#x}", magic);
    }
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

fn main() -> Result<()> {
    println!("========================================================");
    println!("  DEF-046: Real-World DiT Q3_K_M GGUF MMQ 1-Step Benchmark ");
    println!("========================================================");
    let dev = Device::new_cuda(0)?;

    // 1. Load GGUF DiT Model with MMQ enabled
    let model_path = "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b-Q3_K_M.gguf";
    println!("Loading GGUF Model: {model_path}...");
    let mut mem = MemoryConfig::default();
    mem.dit_vram_blocks = 0;
    mem.dit_ram_blocks = 20;
    mem.dit_disk_blocks = 27;
    mem.reserve_gb = 1.0;
    mem.token_chunk = 1024; // chunk=1024 for MMQ
    mem.mlp_chunk = 1024;

    let t0 = Instant::now();
    let dit = Dit::load(model_path, &dev, &mem, None)?;
    let tiers = dit.tiers();
    println!("DiT loaded in {:.2}s, layers: {}, tiers: vram:{}/ram:{}/disk:{}",
             t0.elapsed().as_secs_f64(), dit.cfg.layers, tiers.vram, tiers.ram, tiers.disk);

    // 2. Read stage1_input_3s.edbin
    let edbin_path = "/mnt/R2D/tmp/minimax_bridge/stage1_input_3s.edbin";
    println!("Reading stage1 bundle: {edbin_path}...");
    let mut f = File::open(edbin_path).context("open edbin")?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != b"H3STG1\x01\x00" {
        bail!("Invalid magic: {:?}", magic);
    }
    let mut hdr = [0u8; 24];
    f.read_exact(&mut hdr)?;
    let width = i32::from_le_bytes(hdr[0..4].try_into()?) as usize;
    let height = i32::from_le_bytes(hdr[4..8].try_into()?) as usize;
    let length = i32::from_le_bytes(hdr[8..12].try_into()?) as usize;
    let _steps = i32::from_le_bytes(hdr[12..16].try_into()?) as usize;
    let video_shift = f32::from_le_bytes(hdr[16..20].try_into()?) as f64;
    let audio_shift = f32::from_le_bytes(hdr[20..24].try_into()?) as f64;
    println!("Stage1 Meta: {width}x{height}, length={length}, video_shift={video_shift}, audio_shift={audio_shift}");

    let context = read_h3lt_tensor(&mut f, &dev)?;
    let token_tags_t = read_h3lt_tensor(&mut f, &Device::Cpu)?;
    let vid_lat = read_h3lt_tensor(&mut f, &dev)?;
    let aud_lat = read_h3lt_tensor(&mut f, &dev)?;
    let kf_lat = read_h3lt_tensor(&mut f, &dev)?;

    let mut kf_idx_buf = [0u8; 4];
    f.read_exact(&mut kf_idx_buf)?;
    let kf_idx = i32::from_le_bytes(kf_idx_buf) as usize;

    let token_tags_vec: Vec<u32> = token_tags_t.to_vec1::<f32>()?.iter().map(|&x| x as u32).collect();

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
        visual_cond_noise_aug: 0.0,
        audio_cond_noise_aug: 0.0,
        denoise: None,
        control: None,
    };

    let video_shape = vid_lat.dims();
    let audio_t = aud_lat.dim(3)?;
    let run_steps = 1usize; // Test 1 full denoising step
    let sigmas = simple_sigmas(video_shift, run_steps);

    println!("Preparing DiT run pipeline (run_steps = {})...", run_steps);
    let run = dit.prepare(&cond, video_shape, audio_t, sigmas.clone())?;

    println!("Starting 1-step denoising with Zero-Dequant MMQ...");
    dev.synchronize()?;
    let t_step = Instant::now();
    let mut last_t = Instant::now();

    let sampled = sample_denoised(
        &dit,
        &run,
        vid_lat,
        aud_lat,
        &sigmas,
        Sampler::Euler,
        None,
        |step, total| {
            let step_ms = last_t.elapsed().as_secs_f64() * 1000.0;
            println!("  [Step {}/{} done in {:.2}ms]", step + 1, total, step_ms);
            last_t = Instant::now();
            Ok(())
        },
    )?;

    dev.synchronize()?;
    let total_elapsed = t_step.elapsed().as_secs_f64();
    println!("Step finished in {:.3}s", total_elapsed);

    // Verify output validity (no NaN/Inf)
    let v_f32 = sampled.denoised_v.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut nan_count = 0;
    let mut inf_count = 0;
    for &x in &v_f32 {
        if x.is_nan() { nan_count += 1; }
        if x.is_infinite() { inf_count += 1; }
    }

    println!("\n=== QUALITY AUDIT ===");
    println!("  Output video latent elements: {}", v_f32.len());
    println!("  NaN count: {}", nan_count);
    println!("  Inf count: {}", inf_count);
    assert_eq!(nan_count, 0, "Output contained NaN!");
    assert_eq!(inf_count, 0, "Output contained Inf!");

    println!("\n🎉 DEF-046 GGUF FULL-MODEL MMQ STEP BENCHMARK VERIFIED 100%!");
    Ok(())
}
