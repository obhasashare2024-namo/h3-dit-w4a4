use std::fs::File;
use std::io::Read;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use clap::Parser;
use h3_rs::config::MemoryConfig;
use h3_rs::media::{write_mp4, write_wav};
use h3_rs::vae_audio::AudioVae;
use h3_rs::vae_video::VideoVae;
use h3_rs::weights::Placement;

#[derive(Parser, Debug)]
#[command(name = "pipeline_e2e_av", about = "MiniMax H3 Multi-Modal Render & Decoupled VAE Engine")]
struct Args {
    /// Path to input .edlat file containing video and/or audio latents
    #[arg(short, long, default_value = "/mnt/R2D/tmp/minimax_bridge/stage2_denoised_9006_5s_test.edlat")]
    input: String,

    /// Output media file (.mp4 for video/both, .wav for audio)
    #[arg(short, long, default_value = "/mnt/2tbnvS/tmp/h3_render_output.mp4")]
    output: String,

    /// Target output mode: 'both' (video+audio), 'video' (video only), 'audio' (audio only), 'edlat' (metadata check only, 0 VAE loaded)
    #[arg(short, long, default_value = "both")]
    target: String,

    /// Weight storage placement for Video VAE: 'ram' (system RAM, layer streaming) or 'vram' (pinned in GPU VRAM)
    #[arg(long, default_value = "ram")]
    video_placement: String,

    /// Execution device for Audio VAE: 'cpu' (pure CPU host execution, 0 RTX 3070 VRAM) or 'cuda' (RTX 3070 GPU)
    #[arg(long, default_value = "cpu")]
    audio_device: String,

    /// Execution device for Video VAE: 'cuda' (RTX 3070 GPU) or 'cpu' (pure CPU host execution)
    #[arg(long, default_value = "cuda")]
    video_device: String,

    /// Path to Video VAE weights
    #[arg(long, default_value = "/mnt/2tbnvS/models/vae/minimax_h3_video_vae_int8_convrot.safetensors")]
    video_vae: String,

    /// Path to Audio VAE weights
    #[arg(long, default_value = "/mnt/R2D/models/minimax_h3_audio_vae_fp32.safetensors")]
    audio_vae: String,

    /// Output resolution scaling / interpolation: "none" (native), "1080p" (1920x1088), "2k" (2560x1440), "4k" (3840x2160)
    #[arg(long, default_value = "none")]
    scale: String,

    /// Apply contrast-adaptive detail unsharp filter during interpolation
    #[arg(long, default_value_t = true)]
    unsharp: bool,
}

fn parse_device(name: &str) -> Result<Device> {
    match name.to_lowercase().as_str() {
        "cuda" | "gpu" | "3070" | "vram" => Ok(Device::new_cuda(0)?),
        "cpu" | "ram" | "host" => Ok(Device::Cpu),
        other => bail!("Unknown device: {other}, expected 'cuda'/'3070' or 'cpu'/'ram'"),
    }
}

fn parse_placement(name: &str) -> Result<Placement> {
    match name.to_lowercase().as_str() {
        "ram" | "host" | "cpu" => Ok(Placement::Ram),
        "vram" | "gpu" | "cuda" => Ok(Placement::Gpu),
        other => bail!("Unknown placement: {other}, expected 'ram' or 'vram'"),
    }
}

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
    let args = Args::parse();

    println!("========================================================");
    println!("  MiniMax H3 Multi-Modal Render Engine (Decoupled & E2E)");
    println!("========================================================");
    println!("  Target Mode    : {}", args.target);
    println!("  Input Latent   : {}", args.input);
    println!("  Output Media   : {}", args.output);
    println!("  Video Device   : {} (Placement: {})", args.video_device, args.video_placement);
    println!("  Audio Device   : {} (Pure Host CPU / 0 VRAM Contention)", args.audio_device);
    println!("========================================================");

    let v_dev = parse_device(&args.video_device)?;
    let a_dev = parse_device(&args.audio_device)?;
    let v_place = parse_placement(&args.video_placement)?;

    // 1. Read input latents
    println!("[1] Reading multi-modal denoised latents: {}...", args.input);
    let mut f = File::open(&args.input).context("open edlat")?;

    let zv = read_h3lt_tensor(&mut f, &Device::Cpu)?;
    println!("    Video latent shape: {:?}, dtype: {:?}", zv.dims(), zv.dtype());

    let za_res = read_h3lt_tensor(&mut f, &Device::Cpu);
    let za = match za_res {
        Ok(t) => {
            println!("    Audio latent shape: {:?}, dtype: {:?}", t.dims(), t.dtype());
            Some(t)
        }
        Err(e) => {
            println!("    No second tensor found for audio ({e}).");
            None
        }
    };

    // Mode: edlat only (Zero VAE loaded to VRAM or RAM)
    if args.target == "edlat" {
        println!("\n✅ [EDLAT Mode] Header & Latent metadata verified successfully.");
        println!("   No VAE model loaded to VRAM or RAM (0 byte footprint).");
        return Ok(());
    }

    // Mode: audio only
    if args.target == "audio" {
        let za_tensor = match za {
            Some(t) => t,
            None => bail!("Cannot run audio-only mode: no audio latent in input edlat"),
        };
        println!("\n[Audio Pipeline] Loading Audio VAE to {} from {}...", args.audio_device, args.audio_vae);
        let t_aload = Instant::now();
        let a_vae = AudioVae::load(&args.audio_vae, &a_dev)?;
        println!("    Audio VAE loaded in {:.2}s (sample rate: {} Hz)", t_aload.elapsed().as_secs_f64(), a_vae.sample_rate);

        println!("[Audio Pipeline] Decoding audio waveform (BigVGAN) on {}...", args.audio_device);
        let t_adec = Instant::now();
        let wave = a_vae.decode(&za_tensor.to_device(&a_dev)?)?;
        let sr = a_vae.sample_rate;
        println!("    Audio decoded in {:.2}s on {}, waveform shape: {:?}", t_adec.elapsed().as_secs_f64(), args.audio_device, wave.dims());

        let wav_path = if args.output.ends_with(".wav") {
            args.output.clone()
        } else {
            args.output.replace(".mp4", ".wav")
        };
        write_wav(&wav_path, &wave, sr)?;
        println!("\n🎉 Audio-Only Render Finished Successfully!");
        println!("   Output WAV: {} (Device: {}, 0 RTX 3070 VRAM used)", wav_path, args.audio_device);
        return Ok(());
    }

    // Spawn async concurrent audio decoding thread if target == "both" and audio latent exists
    let audio_thread = if args.target == "both" {
        if let Some(za_tensor) = za {
            let a_dev_clone = a_dev.clone();
            let audio_vae_path = args.audio_vae.clone();
            let audio_device_name = args.audio_device.clone();
            println!("\n[Audio Pipeline] Spawning async concurrent Audio VAE thread on {}...", audio_device_name);
            Some(std::thread::spawn(move || -> Result<(Tensor, u32)> {
                let t_aload = Instant::now();
                let a_vae = AudioVae::load(&audio_vae_path, &a_dev_clone)?;
                println!("    [Async Audio] Audio VAE loaded in {:.2}s (sample rate: {} Hz)", t_aload.elapsed().as_secs_f64(), a_vae.sample_rate);

                println!("    [Async Audio] Decoding audio waveform (BigVGAN) on {}...", audio_device_name);
                let t_adec = Instant::now();
                let wave = a_vae.decode(&za_tensor.to_device(&a_dev_clone)?)?;
                let sr = a_vae.sample_rate;
                println!("    [Async Audio] Audio decoded in {:.2}s on {}, waveform shape: {:?}", t_adec.elapsed().as_secs_f64(), audio_device_name, wave.dims());
                Ok((wave, sr))
            }))
        } else {
            None
        }
    } else {
        None
    };

    // Video pipeline (either 'video' or 'both')
    println!("\n[Video Pipeline] Loading Video VAE (Device: {}, Placement: {:?}) from {}...", args.video_device, v_place, args.video_vae);
    let mut mem = MemoryConfig::default();
    mem.vae_resident = matches!(v_place, Placement::Gpu);
    mem.offload = v_place;
    mem.vae_tile_batch = 1;
    mem.vae_dtype = "f16".into();

    let t_vload = Instant::now();
    let v_vae = VideoVae::load(&args.video_vae, &v_dev, &mem)?;
    println!("    Video VAE loaded in {:.2}s", t_vload.elapsed().as_secs_f64());

    println!("[Video Pipeline] Decoding Video latents to RGB pixels on {}...", args.video_device);
    let zv_target = zv.to_device(&v_dev)?;
    let t_vdec = Instant::now();
    let vid = v_vae.decode_cpuvid(&zv_target)?;
    let (num_frames, h, w) = (vid.f, vid.h, vid.w);
    println!("    Video decoded in {:.2}s, frames: {num_frames}, {w}x{h}, 3 channels @ 24fps (Zero-Copy CpuVid)", t_vdec.elapsed().as_secs_f64());

    // Join async audio thread and synchronize
    let mut audio_tuple: Option<(Tensor, u32)> = None;
    if let Some(handle) = audio_thread {
        println!("\n[Pipeline Sync] Synchronizing concurrent audio decoding thread...");
        let t_sync = Instant::now();
        let (wave, sr) = handle.join().map_err(|e| anyhow::anyhow!("Audio thread panicked: {:?}", e))??;
        println!("    [Pipeline Sync] Audio thread completed! (Wait time: {:.3}s)", t_sync.elapsed().as_secs_f64());
        let expected_samples = ((num_frames as f64 / 24.0) * sr as f64).round() as usize;
        let wave_trimmed = wave.narrow(1, 0, expected_samples.min(wave.dim(1)?))?;
        audio_tuple = Some((wave_trimmed, sr));
    } else if args.target == "both" {
        println!("\n[Video Pipeline] No audio latent found in input edlat, skipping audio mux.");
    } else {
        println!("\n[Video Only Mode] Skipping Audio VAE load & decode (0 audio VRAM used).");
    }

    // Mux output
    let out_mp4 = if args.output.ends_with(".wav") {
        args.output.replace(".wav", ".mp4")
    } else {
        args.output.clone()
    };
    println!("\n[Muxing] Generating MP4: {}...", out_mp4);
    let t_mux = Instant::now();
    let audio_arg = audio_tuple.as_ref().map(|(w, sr)| (w, *sr));
    let scale_opt = if args.scale == "none" || args.scale == "native" {
        None
    } else {
        Some(args.scale.as_str())
    };
    h3_rs::media::write_mp4_from_cpuvid_scaled(&out_mp4, &vid, 24.0, audio_arg, 23, scale_opt, args.unsharp)?;
    println!("    MP4 written in {:.2}s!", t_mux.elapsed().as_secs_f64());

    if let Some((wave, sr)) = audio_tuple {
        let wav_path = out_mp4.replace(".mp4", ".wav");
        let _ = write_wav(&wav_path, &wave, sr);
        println!("    Stand-alone WAV exported: {wav_path}");
    }

    println!("\n🎉 Render Finished Successfully! Mode: {}, Output: {}", args.target, out_mp4);
    Ok(())
}
