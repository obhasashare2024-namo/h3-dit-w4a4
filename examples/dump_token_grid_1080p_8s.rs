use std::fs::File;
use std::io::Read;
use candle_core::{Device, Tensor};
use h3_rs::dit::PackedLayout;

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

fn main() -> anyhow::Result<()> {
    let input_path = "/mnt/2tbnvS/tmp/stage1_input_1080p_8s.edbin";
    println!("=== DEF-080: 1080P 8s Token Grid & Segment Breakdown ===");
    println!("Reading bundle: {}", input_path);

    let mut f = File::open(input_path)?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    let mut header = [0i32; 4];
    for h in &mut header {
        let mut b = [0u8; 4];
        f.read_exact(&mut b)?;
        *h = i32::from_le_bytes(b);
    }
    println!("Header: {:?}", header);

    let mut shifts = [0f32; 2];
    for s in &mut shifts {
        let mut b = [0u8; 4];
        f.read_exact(&mut b)?;
        *s = f32::from_le_bytes(b);
    }
    println!("Shifts: {:?}", shifts);

    let dev = Device::Cpu;
    let context = read_h3lt_tensor(&mut f, &dev)?;
    let token_tags = read_h3lt_tensor(&mut f, &dev)?;
    let vid_lat = read_h3lt_tensor(&mut f, &dev)?;
    let aud_lat = read_h3lt_tensor(&mut f, &dev)?;
    let kf_lat = read_h3lt_tensor(&mut f, &dev)?;

    let text_len = context.dim(0)?;
    let v_dims = vid_lat.dims();
    let a_dims = aud_lat.dims();
    let kf_dims = kf_lat.dims();

    println!("Context (Text) shape: {:?} -> text_len = {}", context.dims(), text_len);
    println!("Token tags shape: {:?}", token_tags.dims());
    println!("Video Latents shape: {:?}", v_dims);
    println!("Audio Latents shape: {:?}", a_dims);
    println!("Keyframe Latents shape: {:?}", kf_dims);

    let (lt, lh, lw) = (v_dims[2], v_dims[3].div_ceil(2) * 2, v_dims[4].div_ceil(2) * 2);
    let audio_t = a_dims[3];

    let kf_specs = vec![h3_rs::dit::KeyframeSpec {
        frame_index: 0,
        video_t: Some(kf_dims[2]),
        audio_t: None,
    }];
    let ref_specs = vec![];

    let layout = PackedLayout::new(text_len, lt, lh, lw, audio_t, &kf_specs, &ref_specs);
    println!("\n=== PackedLayout Results ===");
    println!("Total Sequence Length S = {}", layout.seq_len);
    println!("Segments Breakdown:");
    for (start, stop, kind) in &layout.segments {
        let count = stop - start;
        println!("  [{:6} .. {:6}] ({:6} tokens, {:5.2}%) : {:?}",
            start, stop, count, count as f64 / layout.seq_len as f64 * 100.0, kind);
    }

    let frame_len = (lh / 2) * (lw / 2);
    println!("\nGrid Metrics:");
    println!("  Latent (T, H, W): ({}, {}, {})", lt, lh, lw);
    println!("  Spatial Patch Grid (H/2, W/2): ({}, {}) -> frame_len = {}", lh / 2, lw / 2, frame_len);
    println!("  Target Video Tokens: {} frames * {} patches = {}", lt, frame_len, lt * frame_len);
    println!("  Keyframe Video Tokens: {} frames * {} patches = {}", kf_dims[2], frame_len, kf_dims[2] * frame_len);
    println!("  Target Audio Tokens: {} * 2 = {}", audio_t, audio_t * 2);

    let sum_check = text_len + kf_dims[2] * frame_len + audio_t * 2 + lt * frame_len;
    println!("  Sum check: {} == {} ({})", sum_check, layout.seq_len, sum_check == layout.seq_len);

    Ok(())
}
