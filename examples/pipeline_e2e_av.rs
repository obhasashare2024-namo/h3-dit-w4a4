use std::fs::File;
use std::io::Read;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use h3_rs::config::MemoryConfig;
use h3_rs::media::{write_mp4, write_wav};
use h3_rs::vae_audio::AudioVae;
use h3_rs::vae_video::VideoVae;

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
    let args: Vec<String> = std::env::args().collect();
    let edlat_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "/mnt/R2D/tmp/minimax_bridge/stage2_denoised_9006_5s_test.edlat".to_string()
    };
    let out_mp4 = if args.len() > 2 {
        args[2].clone()
    } else {
        "/mnt/2tbnvS/tmp/h3_render_e2e_audio_video_5s.mp4".to_string()
    };
    let video_vae_path = if args.len() > 3 {
        args[3].clone()
    } else {
        "/mnt/2tbnvS/models/vae/minimax_h3_video_vae_int8_convrot.safetensors".to_string()
    };
    let audio_vae_path = if args.len() > 4 {
        args[4].clone()
    } else {
        "/mnt/R2D/models/minimax_h3_audio_vae_fp32.safetensors".to_string()
    };

    println!("========================================================");
    println!("  MiniMax H3 Pure Rust E2E Video+Audio Render (Mode E)  ");
    println!("========================================================");
    let dev = Device::new_cuda(0)?;

    println!("[1] Reading multi-modal denoised latents: {edlat_path}...");
    let mut f = File::open(&edlat_path).context("open edlat")?;
    let zv = read_h3lt_tensor(&mut f, &dev)?;
    println!("    Video latent shape: {:?}, dtype: {:?}", zv.dims(), zv.dtype());

    let za_res = read_h3lt_tensor(&mut f, &dev);
    let za = match za_res {
        Ok(t) => {
            println!("    Audio latent shape: {:?}, dtype: {:?}", t.dims(), t.dtype());
            Some(t)
        }
        Err(e) => {
            println!("    No second tensor found for audio ({e}), rendering video only.");
            None
        }
    };

    // 2. Load Video VAE (Layer-major streaming from local NVMe)
    println!("[2] Loading Video VAE from: {video_vae_path}...");
    let mut mem = MemoryConfig::default();
    mem.vae_resident = false;
    mem.vae_tile_batch = 1;
    mem.vae_dtype = "f16".into();
    mem.offload = h3_rs::weights::Placement::Ram;

    let t_vload = Instant::now();
    let v_vae = VideoVae::load(&video_vae_path, &dev, &mem)?;
    println!("    Video VAE loaded in {:.2}s", t_vload.elapsed().as_secs_f64());

    // 3. Decode Video Latents
    println!("[3] Decoding Video latents to RGB pixels...");
    let t_vdec = Instant::now();
    let px = v_vae.decode(&zv)?;
    println!("    Video decoded in {:.2}s, raw shape: {:?}", t_vdec.elapsed().as_secs_f64(), px.dims());
    let frames = px.permute((1, 0, 2, 3))?.contiguous()?;
    let (num_frames, c, h, w) = frames.dims4()?;
    println!("    Video frames: {num_frames}, {w}x{h}, {c} channels @ 24fps");

    // 4. Decode Audio Latents (if present)
    let mut audio_tuple: Option<(Tensor, u32)> = None;
    if let Some(za_tensor) = za {
        println!("[4] Loading Audio VAE from: {audio_vae_path}...");
        let t_aload = Instant::now();
        let a_vae = AudioVae::load(&audio_vae_path, &dev)?;
        println!("    Audio VAE loaded in {:.2}s (sample rate: {} Hz)", t_aload.elapsed().as_secs_f64(), a_vae.sample_rate);

        println!("[5] Decoding Audio waveform (BigVGAN)...");
        let t_adec = Instant::now();
        let wave = a_vae.decode(&za_tensor)?;
        let sr = a_vae.sample_rate;
        println!("    Audio decoded in {:.2}s, waveform shape: {:?}", t_adec.elapsed().as_secs_f64(), wave.dims());

        // Align duration with video frames (24 fps)
        let expected_samples = ((num_frames as f64 / 24.0) * sr as f64).round() as usize;
        let wave_trimmed = wave.narrow(1, 0, expected_samples.min(wave.dim(1)?))?;
        println!("    Aligned audio samples: {} ({:.2}s)", wave_trimmed.dim(1)?, wave_trimmed.dim(1)? as f64 / sr as f64);
        audio_tuple = Some((wave_trimmed, sr));
    }

    // 5. Mux Video + Audio to MP4
    println!("[6] Muxing Video + Audio to final MP4: {out_mp4}...");
    let t_mux = Instant::now();
    let audio_arg = audio_tuple.as_ref().map(|(w, sr)| (w, *sr));
    write_mp4(&out_mp4, &frames, 24.0, audio_arg, 23)?;
    println!("    MP4 muxed successfully in {:.2}s!", t_mux.elapsed().as_secs_f64());

    if let Some((wave, sr)) = audio_tuple {
        let wav_path = out_mp4.replace(".mp4", ".wav");
        let _ = write_wav(&wav_path, &wave, sr);
        println!("    Stand-alone WAV exported: {wav_path}");
    }

    println!("\n🎉 Pure Rust Mode E (Video + Audio) Finished Successfully!");
    println!("   Output File: {out_mp4}");
    Ok(())
}
