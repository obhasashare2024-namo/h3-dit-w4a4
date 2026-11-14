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

    // Reverse GGML shape order [W, H, ...] to C/PyTorch row-major layout [..., H, W]
    shape.reverse();
    Ok(Tensor::from_vec(data, shape, dev)?)
}

fn main() -> Result<()> {
    println!("=== MiniMax H3 W4A4 DiT End-to-End 3s Sampling Benchmark (Port 9411) ===");
    let dev = Device::new_cuda(0)?;
    
    // 1. Load W4A4 DiT Model
    let model_path = "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.safetensors";
    println!("Loading W4A4 Model: {model_path}...");
    let mut mem = MemoryConfig::default();
    let requested_blocks = std::env::var("H3_BLOCKS").ok().and_then(|s| s.parse::<i64>().ok()).unwrap_or(-1);
    mem.dit_gpu_blocks = requested_blocks;
    mem.reserve_gb = 1.0;
    
    let t0 = Instant::now();
    let dit = Dit::load(model_path, &dev, &mem, None)?;
    println!("DiT loaded in {:.2}s, resident blocks: {}/{}", t0.elapsed().as_secs_f64(), dit.resident_blocks(), dit.cfg.layers);

    // 2. Read stage1_input_3s.edbin
    let edbin_path = "/mnt/R2D/tmp/minimax_bridge/stage1_input_3s.edbin";
    println!("Reading stage1 bundle: {edbin_path}...");
    let mut f = File::open(edbin_path).context("open edbin")?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != b"H3STG1\x01\x00" {
        bail!("Invalid magic: {:?}", magic);
    }
    let mut hdr = [0u8; 24]; // 4 i32, 2 f32
    f.read_exact(&mut hdr)?;
    let width = i32::from_le_bytes(hdr[0..4].try_into()?) as usize;
    let height = i32::from_le_bytes(hdr[4..8].try_into()?) as usize;
    let length = i32::from_le_bytes(hdr[8..12].try_into()?) as usize;
    let steps = i32::from_le_bytes(hdr[12..16].try_into()?) as usize;
    let video_shift = f32::from_le_bytes(hdr[16..20].try_into()?) as f64;
    let audio_shift = f32::from_le_bytes(hdr[20..24].try_into()?) as f64;
    println!("Stage1 Meta: {width}x{height}, length={length}, steps={steps}, video_shift={video_shift}, audio_shift={audio_shift}");

    let context = read_h3lt_tensor(&mut f, &dev)?;
    let token_tags_t = read_h3lt_tensor(&mut f, &Device::Cpu)?;
    let vid_lat = read_h3lt_tensor(&mut f, &dev)?;
    let aud_lat = read_h3lt_tensor(&mut f, &dev)?;
    let kf_lat = read_h3lt_tensor(&mut f, &dev)?;
    
    let mut kf_idx_buf = [0u8; 4];
    f.read_exact(&mut kf_idx_buf)?;
    let kf_idx = i32::from_le_bytes(kf_idx_buf) as usize;

    println!("Context shape: {:?}, VidLat shape: {:?}, AudLat shape: {:?}, KfLat: {:?}, KfIdx: {kf_idx}",
             context.dims(), vid_lat.dims(), aud_lat.dims(), kf_lat.dims());

    // 3. Prepare inputs
    let token_tags_vec: Vec<u32> = token_tags_t.to_vec1::<f32>()?.iter().map(|&x| x as u32).collect();
    
    let xv = vid_lat;
    let xa = aud_lat;
    let kf_v = kf_lat;

    let cond = DitCond {
        text: context,
        tags: Some(token_tags_vec),
        keyframes: vec![KeyframeCond {
            frame_index: kf_idx,
            video: Some(kf_v),
            audio: None,
        }],
        refs: vec![],
        seed: 42,
        visual_cond_noise_aug: 0.0,
        audio_cond_noise_aug: 0.0,
        denoise: None,
        control: None,
    };

    let video_shape = xv.dims();
    let audio_t = xa.dim(3)?;
    let sigmas = simple_sigmas(video_shift, steps);

    println!("Preparing DiT run (video_shape={video_shape:?}, audio_t={audio_t}, sigmas={sigmas:?})...");
    let t_prep = Instant::now();
    let run = dit.prepare(&cond, video_shape, audio_t, sigmas.clone())?;
    println!("DiT prepared in {:.2}ms", t_prep.elapsed().as_secs_f64() * 1000.0);

    // 4. Run End-to-End Euler Sampling
    println!("\n>>> STARTING 6-STEP W4A4 EULER SAMPLING BENCHMARK <<<");
    let t_sample = Instant::now();
    let mut last_t = Instant::now();
    let sampled = sample_denoised(
        &dit,
        &run,
        xv,
        xa,
        &sigmas,
        Sampler::Euler,
        None,
        |step, total| {
            let step_ms = last_t.elapsed().as_secs_f64() * 1000.0;
            println!("  [Step {step}/{total}] Euler step finished in {:.2}ms (Elapsed: {:.2}s)", step_ms, t_sample.elapsed().as_secs_f64());
            last_t = Instant::now();
            Ok(())
        }
    )?;
    let total_sample_sec = t_sample.elapsed().as_secs_f64();
    println!("\n>>> 6-STEP SAMPLING COMPLETED in {:.3}s ({:.2}s / step) <<<", total_sample_sec, total_sample_sec / steps as f64);

    // 5. Check Non-zero & Non-NaN
    let v_flat = sampled.denoised_v.flatten_all()?.to_vec1::<f32>()?;
    let _a_flat = sampled.denoised_a.flatten_all()?.to_vec1::<f32>()?;
    let mut nan_count = 0;
    let mut zero_count = 0;
    for &x in &v_flat {
        if x.is_nan() { nan_count += 1; }
        if x == 0.0 { zero_count += 1; }
    }
    println!("Sanity Check: Denoised Video elements={}, NaNs={}, Zeroes={}", v_flat.len(), nan_count, zero_count);
    assert_eq!(nan_count, 0, "Video output contains NaNs!");

    // 6. Export to .edlat and save
    let out_edlat = "/mnt/R2D/tmp/minimax_bridge/stage2_denoised_w4a4_9411.edlat";
    println!("Writing denoised latents to {out_edlat}...");
    let mut out_f = File::create(out_edlat)?;
    let mut write_edlat_tensor = |t: &Tensor| -> Result<()> {
        use std::io::Write;
        // In .edlat format: write shape in GGML order [W, H, ...]
        let mut dims = t.dims().to_vec();
        dims.reverse();
        let numel = t.elem_count() as u64;
        let mut shape8 = [0i64; 8];
        for (i, &d) in dims.iter().enumerate() {
            shape8[i] = d as i64;
        }
        let magic = 0x48334C54u32; // H3LT
        out_f.write_all(&magic.to_le_bytes())?;
        out_f.write_all(&(dims.len() as u32).to_le_bytes())?;
        for &s in &shape8 {
            out_f.write_all(&s.to_le_bytes())?;
        }
        out_f.write_all(&numel.to_le_bytes())?;
        let data = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        out_f.write_all(bytes)?;
        Ok(())
    };

    write_edlat_tensor(&sampled.denoised_v)?;
    write_edlat_tensor(&sampled.denoised_a)?;
    println!("Successfully saved {out_edlat}!");

    println!("=== W4A4 DiT BENCHMARK SUCCESSFUL! ===");
    Ok(())
}
