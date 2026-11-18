use std::fs::File;
use std::io::Read;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use h3_rs::config::MemoryConfig;
use h3_rs::vae_video::VideoVae;
use h3_rs::media::write_mp4;

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
        "/mnt/R2D/tmp/minimax_bridge/stage2_denoised_1080p_3s.edlat".to_string()
    };
    let out_mp4 = if args.len() > 2 {
        args[2].clone()
    } else {
        "/mnt/2tbnvS/tmp/h3_render_1080p_3s_noaudio.mp4".to_string()
    };

    println!("========================================================");
    println!("  MiniMax H3 Pure Rust Video Render from .edlat         ");
    println!("========================================================");
    let dev = Device::new_cuda(0)?;

    println!("[1] Reading denoised latent: {edlat_path}...");
    let mut f = File::open(&edlat_path).context("open edlat")?;
    let z = read_h3lt_tensor(&mut f, &dev)?;
    println!("    Latent shape: {:?}, dtype: {:?}", z.dims(), z.dtype());

    // 2. Configure VAE for 8GB GPU (streaming blocks to save VRAM)
    let vae_path = if args.len() > 3 { args[3].clone() } else { "/mnt/2tbnvS/models/vae/minimax_h3_video_vae_int8_convrot.safetensors".to_string() };
    println!("[2] Loading Video VAE from: {vae_path}...");
    let mut mem = MemoryConfig::default();
    mem.vae_resident = false;      // stream blocks from host to ensure 0 VRAM blowout
    mem.vae_tile_batch = 1;        // tile batch 1
    mem.vae_dtype = "f16".into();  // FP16
    mem.offload = h3_rs::weights::Placement::Ram;

    let t0 = Instant::now();
    let vae = VideoVae::load(&vae_path, &dev, &mem)?;
    println!("    Video VAE loaded in {:.2}s", t0.elapsed().as_secs_f64());

    // 3. Decode latents to video pixels
    println!("[3] Decoding latents to RGB pixels (3D ViT Spatial Tiling + Temporal Chunking)...");
    let t_dec = Instant::now();
    let px = vae.decode(&z)?; // returns [3, F, H, W]
    println!("    Pixels decoded in {:.2}s, raw shape: {:?}", t_dec.elapsed().as_secs_f64(), px.dims());

    // 4. Permute to [F, 3, H, W] for write_mp4
    let frames = px.permute((1, 0, 2, 3))?.contiguous()?;
    let (num_frames, c, h, w) = frames.dims4()?;
    println!("    Rendered video: {num_frames} frames, {w}x{h}, {c} channels");

    // 5. Write to MP4 via ffmpeg (zero audio)
    println!("[4] Muxing pure video to MP4 (no audio): {out_mp4}...");
    let t_mux = Instant::now();
    write_mp4(&out_mp4, &frames, 24.0, None, 23)?;
    println!("    MP4 muxed successfully in {:.2}s!", t_mux.elapsed().as_secs_f64());

    println!("\n🎉 Pure Rust Video Render Finished Successfully!");
    println!("   Output File: {out_mp4}");
    Ok(())
}
