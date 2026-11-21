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
    // Explicitly select fastpath Tensor Core Dequant Arena (crush 3157s bottleneck)
    std::env::set_var("H3_MMQ", "0");

    let dev = Device::new_cuda(0)?;
    let model_path = "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b-Q3_K_M.gguf";
    let mut mem = MemoryConfig::default();
    mem.dit_vram_blocks = 0;
    mem.dit_ram_blocks = 48;
    mem.dit_disk_blocks = 0;
    mem.offload = h3_rs::weights::Placement::Ram;

    println!("[gguf-q3km-e2e] Loading GGUF Q3_K_M DiT with Zero-Disk Placement: {}", model_path);
    let dit = Dit::load(model_path, &dev, &mem, None)?;

    let args: Vec<String> = std::env::args().collect();
    let input_path = if args.len() > 1 { args[1].clone() } else { "/mnt/2tbnvS/tmp/stage1_input_1080p_8s.edbin".to_string() };
    let out_path = if args.len() > 2 { args[2].clone() } else { "/mnt/2tbnvS/tmp/h3_q3km_e2e_1080p_8s.edlat".to_string() };
    println!("[gguf-q3km-e2e] Input bundle: {}", input_path);
    println!("[gguf-q3km-e2e] Output edlat: {}", out_path);
    let mut f = File::open(&input_path)?;
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

    println!("[gguf-q3km-e2e] Preparing DiT run with {} sigmas: {:?}", sigmas.len(), sigmas);
    let run = dit.prepare(&cond, &video_shape, audio_t, sigmas.clone())?;

    println!("[gguf-q3km-e2e] Running full 6-step sampling (Rank 1 OP-1 Accelerated)...");
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
            println!("[gguf-q3km-e2e] Step {}/{} completed in {:.2}ms ({:.2}s)", step + 1, total, step_ms, step_ms / 1000.0);
            last_t = Instant::now();
            Ok(())
        },
    )?;
    let total_elapsed = t0.elapsed().as_secs_f64();
    println!("[gguf-q3km-e2e] All 6 steps finished in {:.2}s (avg {:.2}s/step)", total_elapsed, total_elapsed / 6.0);

    // out_path handled above
    let mut out_file = File::create(&out_path)?;
    write_edlat_tensor(&mut out_file, &sampled.denoised_v)?;
    write_edlat_tensor(&mut out_file, &sampled.denoised_a)?;
    println!("[gguf-q3km-e2e] Saved multi-modal sampled latents to {out_path}");

    Ok(())
}
