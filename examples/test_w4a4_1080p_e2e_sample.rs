use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;
use candle_core::{DType, Device, Tensor};
use h3_rs::config::MemoryConfig;
use h3_rs::dit::{Dit, DitCond, KeyframeCond};
use h3_rs::sampling::{sample_denoised, simple_sigmas, Sampler};

fn read_h3lt_tensor<R: Read>(r: &mut R, dev: &Device) -> anyhow::Result<Tensor> {
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

fn write_edlat_tensor<W: Write>(w: &mut W, t: &Tensor) -> anyhow::Result<()> {
    let dims = t.dims();
    let numel = t.elem_count();
    let mut hdr = [0u8; 80];
    hdr[0..4].copy_from_slice(&0x48334c54u32.to_le_bytes());
    hdr[4..8].copy_from_slice(&(dims.len() as u32).to_le_bytes());
    let mut rev_dims = dims.to_vec();
    rev_dims.reverse();
    for (i, &d) in rev_dims.iter().enumerate() {
        hdr[8 + i * 8..16 + i * 8].copy_from_slice(&(d as i64).to_le_bytes());
    }
    hdr[72..80].copy_from_slice(&(numel as u64).to_le_bytes());
    w.write_all(&hdr)?;
    let data_f32 = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let slice_u8 = unsafe {
        std::slice::from_raw_parts(data_f32.as_ptr() as *const u8, data_f32.len() * 4)
    };
    w.write_all(slice_u8)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    println!("========================================================");
    println!("  MiniMax H3 W4A4 1080P 3s End-to-End Portrait Sampling ");
    println!("========================================================");

    let dev = Device::new_cuda(0)?;
    let model_path = "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.safetensors";
    let input_bundle = "/mnt/R2D/tmp/minimax_bridge/stage1_input_1080p_3s.edbin";

    let mut mem = MemoryConfig::default();
    mem.dit_gpu_blocks = 26;
    mem.dit_vram_blocks = 26;
    mem.dit_ram_blocks = 21;
    mem.dit_disk_blocks = 0;
    mem.token_chunk = 4096;
    mem.mlp_chunk = 2048;
    mem.offload = h3_rs::weights::Placement::Disk;

    println!("[w4a4-1080p] Loading W4A4 DiT...");
    let dit = Dit::load(model_path, &dev, &mem, None)?;

    println!("[w4a4-1080p] Reading 1080P 3s bundle: {input_bundle}...");
    let mut f = File::open(input_bundle)?;
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

    let context = read_h3lt_tensor(&mut f, &dev)?;
    let token_tags_t = read_h3lt_tensor(&mut f, &Device::Cpu)?;
    let vid_lat = read_h3lt_tensor(&mut f, &dev)?;
    let aud_lat = read_h3lt_tensor(&mut f, &dev)?;
    let kf_lat = read_h3lt_tensor(&mut f, &dev)?;
    let mut kf_idx_bytes = [0u8; 4];
    f.read_exact(&mut kf_idx_bytes)?;
    let kf_idx = i32::from_le_bytes(kf_idx_bytes) as usize;

    let token_tags_vec: Vec<u32> = token_tags_t.to_vec1::<f32>()?.iter().map(|&x| {
        if x > 0.0 && x < 1e-30 { x.to_bits() } else { x as u32 }
    }).collect();

    println!("[w4a4-1080p] Video Latent Shape: {:?}, Audio Latent Shape: {:?}", vid_lat.dims(), aud_lat.dims());

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
    let sigmas = dit.hyperflow_sigmas.clone().unwrap_or_else(|| simple_sigmas(shifts[0] as f64, 6));

    let mut gen = h3_rs::rng::TorchGen::new(42);
    let nv = gen.randn_tensor(&video_shape, &dev)?;
    let na = gen.randn_tensor(&[1, 32, 2, audio_t], &dev)?;
    let xv = (&nv * sigmas[0])?;
    let xa = (&na * sigmas[0])?;

    println!("[w4a4-1080p] Preparing DiT run with {} sigmas: {:?}", sigmas.len(), sigmas);
    let run = dit.prepare(&cond, &video_shape, audio_t, sigmas.clone())?;

    println!("[w4a4-1080p] Running full 6-step 1080P sampling...");
    let t0 = Instant::now();
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
            println!("[w4a4-1080p] Step {}/{} completed in {:.2}ms ({:.2}s)", step, total, step_ms, step_ms / 1000.0);
            last_t = Instant::now();
            Ok(())
        },
    )?;
    println!("[w4a4-1080p] All 6 steps finished in {:.2}s", t0.elapsed().as_secs_f64());

    let out_path = "/mnt/2tbnvS/tmp/h3_w4a4_1080p_3s.edlat";
    let mut out_file = File::create(out_path)?;
    write_edlat_tensor(&mut out_file, &sampled.denoised_v)?;
    write_edlat_tensor(&mut out_file, &sampled.denoised_a)?;
    println!("[w4a4-1080p] Saved 1080P multi-modal sampled latents to {out_path} ({:.2} MB)", std::fs::metadata(out_path)?.len() as f64 / 1e6);

    Ok(())
}
