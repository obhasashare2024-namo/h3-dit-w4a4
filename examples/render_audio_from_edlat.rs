use std::fs::File;
use std::io::Read;
use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use h3_rs::vae_audio::AudioVae;
use h3_rs::media::write_wav;

fn read_h3lt_tensor<R: Read>(r: &mut R, dev: &Device) -> Result<Option<Tensor>> {
    let mut hdr = [0u8; 80];
    if r.read_exact(&mut hdr).is_err() {
        return Ok(None);
    }
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
    Ok(Some(Tensor::from_vec(data, shape, dev)?))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: render_audio_from_edlat <in.edlat> <out.wav>");
        std::process::exit(1);
    }
    let edlat_path = &args[1];
    let out_wav = &args[2];
    let dev = Device::new_cuda(0)?;

    println!("========================================================");
    println!("  MiniMax H3 Audio Render from .edlat                   ");
    println!("========================================================");
    println!("[1] Reading latents from: {edlat_path}...");
    let mut f = File::open(edlat_path).context("open edlat")?;
    let _vid = read_h3lt_tensor(&mut f, &dev)?.context("missing video tensor")?;
    let aud = read_h3lt_tensor(&mut f, &dev)?.context("missing audio tensor")?;
    println!("    Audio Latent shape: {:?}, dtype: {:?}", aud.dims(), aud.dtype());

    let vae_path = "/mnt/R2D/models/minimax_h3_audio_vae_fp32.safetensors";
    println!("[2] Loading Audio VAE from: {vae_path}...");
    let avae = AudioVae::load(vae_path, &dev)?;

    println!("[3] Decoding audio latent...");
    let wave = avae.decode(&aud)?;
    println!("    Decoded wave shape: {:?}, dtype: {:?}", wave.dims(), wave.dtype());

    println!("[4] Writing output wav to: {out_wav}...");
    write_wav(out_wav, &wave, 32000)?;
    println!("Done! Output saved to: {out_wav}");
    Ok(())
}
