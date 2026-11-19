use std::fs::File;
use std::io::Read;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
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
    println!("  GGUF 14B Acceleration Probe: 1-Step Denoise Benchmark ");
    println!("========================================================");

    let dev = Device::new_cuda(0)?;
    let model_path = "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b-Q3_K_M.gguf";

    // 1. Configure Zero-Disk Memory Placement (RAM: 48, VRAM: 0, Disk: 0)
    let mut mem = MemoryConfig::default();
    mem.dit_vram_blocks = 0;
    mem.dit_ram_blocks = 48;
    mem.dit_disk_blocks = 0;
    mem.offload = h3_rs::weights::Placement::Ram;

    println!("Loading GGUF DiT Model from: {}", model_path);
    let t_load = Instant::now();
    let dit = Dit::load(model_path, &dev, &mem, None)?;
    let tiers = dit.tiers();
    println!("DiT loaded in {:.2}s, tiers: vram:{}/ram:{}/disk:{}",
             t_load.elapsed().as_secs_f64(), tiers.vram, tiers.ram, tiers.disk);

    // 2. Read stage1 bundle
    let edbin_path = "/mnt/R2D/tmp/minimax_bridge/stage1_input_3s.edbin";
    println!("Reading stage1 bundle: {}", edbin_path);
    let mut f = File::open(edbin_path).context("open edbin")?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    let mut hdr = [0u8; 24];
    f.read_exact(&mut hdr)?;
    let width = i32::from_le_bytes(hdr[0..4].try_into()?) as usize;
    let height = i32::from_le_bytes(hdr[4..8].try_into()?) as usize;
    let length = i32::from_le_bytes(hdr[8..12].try_into()?) as usize;
    let video_shift = f32::from_le_bytes(hdr[16..20].try_into()?) as f64;
    println!("Stage1 Meta: {}x{}, length={}, shift={}", width, height, length, video_shift);

    let context = read_h3lt_tensor(&mut f, &dev)?;
    let token_tags_t = read_h3lt_tensor(&mut f, &Device::Cpu)?;
    let vid_lat = read_h3lt_tensor(&mut f, &dev)?;
    let aud_lat = read_h3lt_tensor(&mut f, &dev)?;
    let kf_lat = read_h3lt_tensor(&mut f, &dev)?;
    let mut kf_idx_buf = [0u8; 4];
    f.read_exact(&mut kf_idx_buf)?;
    let kf_idx = i32::from_le_bytes(kf_idx_buf) as usize;

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

    let video_shape = vid_lat.dims();
    let audio_t = aud_lat.dim(3)?;
    let sigmas = simple_sigmas(video_shift, 1);

    let mut gen = h3_rs::rng::TorchGen::new(42);
    let nv = gen.randn_tensor(video_shape, &dev)?;
    let na = gen.randn_tensor(&[1, 32, 2, audio_t], &dev)?;
    let xv = (&nv * sigmas[0])?;
    let xa = (&na * sigmas[0])?;

    println!("Preparing DiT run pipeline (1 step)... ");
    let run = dit.prepare(&cond, video_shape, audio_t, sigmas.clone())?;

    println!("Starting 1-step test...");
    dev.synchronize()?;
    let t_step = Instant::now();

    let _sampled = sample_denoised(
        &dit,
        &run,
        xv,
        xa,
        &sigmas,
        Sampler::Euler,
        None,
        |step, total| {
            println!("Step {}/{} completed in {:.2}ms", step + 1, total, t_step.elapsed().as_secs_f64() * 1000.0);
            Ok(())
        },
    )?;
    dev.synchronize()?;
    let total_s = t_step.elapsed().as_secs_f64();
    println!("========================================================");
    println!("  Total 1-step latency: {:.2}s", total_s);
    println!("  Projected 6-step latency: {:.2}s", total_s * 6.0);
    println!("========================================================");

    Ok(())
}
