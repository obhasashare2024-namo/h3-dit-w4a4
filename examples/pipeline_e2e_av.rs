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

    /// Weight storage placement for Video VAE: 'ram' (system RAM, layer-by-layer streaming to GPU, default for low VRAM), 'vram' (pinned in VRAM), or 'disk' (mmap)
    #[arg(long, default_value = "ram")]
    video_placement: String,

    /// Compute device for Video VAE: 'cuda' (GPU)
    #[arg(long, default_value = "cuda")]
    video_device: String,

    /// Compute device for Audio VAE: 'cuda' (GPU)
    #[arg(long, default_value = "cuda")]
    audio_device: String,

    /// Path to Video VAE weights
    #[arg(long, default_value = "/mnt/2tbnvS/models/vae/minimax_h3_video_vae_int8_convrot.safetensors")]
    video_vae: String,

    /// Path to Audio VAE weights
    #[arg(long, default_value = "/mnt/R2D/models/minimax_h3_audio_vae_fp32.safetensors")]
    audio_vae: String,
}

fn parse_device(name: &str) -> Result<Device> {
    match name.to_lowercase().as_str() {
        "cuda" | "gpu" | "vram" => Ok(Device::new_cuda(0)?),
        "cpu" => Ok(Device::Cpu),
        other => bail!("Unknown device: {other}, expected 'cuda' or 'cpu'"),
    }
}

fn parse_placement(name: &str) -> Result<Placement> {
    match name.to_lowercase().as_str() {
        "ram" | "host" => Ok(Placement::Ram),
        "vram" | "gpu" => Ok(Placement::Gpu),
        "disk" | "mmap" => Ok(Placement::Disk),
        other => bail!("Unknown placement: {other}, expected 'ram', 'vram', or 'disk'"),
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
    println!("  Target Mode   : {}", args.target);
    println!("  Input Latent  : {}", args.input);
    println!("  Output Media  : {}", args.output);
    println!("  Video Storage : {}", args.video_placement);
    println!("  Video Device  : {}", args.video_device);
    println!("  Audio Device  : {}", args.audio_device);
    println!("========================================================");

    let v_dev = parse_device(&args.video_device)?;
    let a_dev = parse_device(&args.audio_device)?;
    let v_place = parse_placement(&args.video_placement)?;

    // 1. Read input latents
    println!("[1] Reading multi-modal denoised latents: {}...", args.input);
    let mut f = File::open(&args.input).context("open edlat")?;

    let load_dev = if args.target == "edlat" || args.target == "audio" {
        &Device::Cpu
    } else {
        &v_dev
    };

    let zv = read_h3lt_tensor(&mut f, load_dev)?;
    println!("    Video latent shape: {:?}, dtype: {:?}", zv.dims(), zv.dtype());

    let za_res = read_h3lt_tensor(&mut f, if args.target == "audio" { &a_dev } else { &Device::Cpu });
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
        println!("
✅ [EDLAT Mode] Header & Latent metadata verified successfully.");
        println!("   No VAE model loaded to VRAM or RAM (0 byte footprint).");
        return Ok(());
    }

    // Mode: audio only
    if args.target == "audio" {
        let za_tensor = match za {
            Some(t) => t.to_device(&a_dev)?,
            None => bail!("Cannot run audio-only mode: no audio latent in input edlat"),
        };
        println!("
[Audio Pipeline] Loading Audio VAE to {} from {}...", args.audio_device, args.audio_vae);
        let t_aload = Instant::now();
        let a_vae = AudioVae::load(&args.audio_vae, &a_dev)?;
        println!("    Audio VAE loaded in {:.2}s (sample rate: {} Hz)", t_aload.elapsed().as_secs_f64(), a_vae.sample_rate);

        println!("[Audio Pipeline] Decoding audio waveform (BigVGAN)...");
        let t_adec = Instant::now();
        let wave = a_vae.decode(&za_tensor)?;
        let sr = a_vae.sample_rate;
        println!("    Audio decoded in {:.2}s, waveform shape: {:?}", t_adec.elapsed().as_secs_f64(), wave.dims());

        let wav_path = if args.output.ends_with(".wav") {
            args.output.clone()
        } else {
            args.output.replace(".mp4", ".wav")
        };
        write_wav(&wav_path, &wave, sr)?;
        println!("
🎉 Audio-Only Render Finished Successfully!");
        println!("   Output WAV: {wav_path} (No Video VAE loaded, 0 video VRAM used)");
        return Ok(());
    }

    // Video pipeline (either 'video' or 'both')
    println!("
[Video Pipeline] Loading Video VAE (Placement: {:?}) from {}...", v_place, args.video_vae);
    let mut mem = MemoryConfig::default();
    mem.vae_resident = matches!(v_place, Placement::Gpu);
    mem.offload = v_place;
    mem.vae_tile_batch = 1;
    mem.vae_dtype = "f16".into();

    let t_vload = Instant::now();
    let v_vae = VideoVae::load(&args.video_vae, &v_dev, &mem)?;
    println!("    Video VAE loaded in {:.2}s", t_vload.elapsed().as_secs_f64());

    println!("[Video Pipeline] Decoding Video latents to RGB pixels...");
    let zv_target = zv.to_device(&v_dev)?;
    let t_vdec = Instant::now();
    let px = v_vae.decode(&zv_target)?;
    println!("    Video decoded in {:.2}s, raw shape: {:?}", t_vdec.elapsed().as_secs_f64(), px.dims());
    let frames = px.permute((1, 0, 2, 3))?.contiguous()?;
    let (num_frames, c, h, w) = frames.dims4()?;
    println!("    Video frames: {num_frames}, {w}x{h}, {c} channels @ 24fps");

    // Check if we need audio
    let mut audio_tuple: Option<(Tensor, u32)> = None;
    if args.target == "both" {
        if let Some(za_tensor) = za {
            let za_target = za_tensor.to_device(&a_dev)?;
            println!("
[Audio Pipeline] Loading Audio VAE to {} from {}...", args.audio_device, args.audio_vae);
            let t_aload = Instant::now();
            let a_vae = AudioVae::load(&args.audio_vae, &a_dev)?;
            println!("    Audio VAE loaded in {:.2}s (sample rate: {} Hz)", t_aload.elapsed().as_secs_f64(), a_vae.sample_rate);

            println!("[Audio Pipeline] Decoding Audio waveform (BigVGAN)...");
            let t_adec = Instant::now();
            let wave = a_vae.decode(&za_target)?;
            let sr = a_vae.sample_rate;
            println!("    Audio decoded in {:.2}s, waveform shape: {:?}", t_adec.elapsed().as_secs_f64(), wave.dims());

            let expected_samples = ((num_frames as f64 / 24.0) * sr as f64).round() as usize;
            let wave_trimmed = wave.narrow(1, 0, expected_samples.min(wave.dim(1)?))?;
            audio_tuple = Some((wave_trimmed, sr));
        }
    } else {
        println!("
[Video Only Mode] Skipping Audio VAE load & decode (0 audio VRAM used).");
    }

    // Mux output
    let out_mp4 = if args.output.ends_with(".wav") {
        args.output.replace(".wav", ".mp4")
    } else {
        args.output.clone()
    };
    println!("
[Muxing] Generating MP4: {out_mp4}...");
    let t_mux = Instant::now();
    let audio_arg = audio_tuple.as_ref().map(|(w, sr)| (w, *sr));
    write_mp4(&out_mp4, &frames, 24.0, audio_arg, 23)?;
    println!("    MP4 written in {:.2}s!", t_mux.elapsed().as_secs_f64());

    if let Some((wave, sr)) = audio_tuple {
        let wav_path = out_mp4.replace(".mp4", ".wav");
        let _ = write_wav(&wav_path, &wave, sr);
        println!("    Stand-alone WAV exported: {wav_path}");
    }

    println!("
🎉 Render Finished Successfully! Mode: {}, Output: {}", args.target, out_mp4);
    Ok(())
}
