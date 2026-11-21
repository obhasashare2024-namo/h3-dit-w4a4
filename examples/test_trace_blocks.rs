use std::fs::File;
use std::io::Read;
use candle_core::{DType, Device, Tensor};
use h3_rs::config::MemoryConfig;
use h3_rs::dit::{Dit, DitCond, KeyframeCond};

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

fn cos_sim(a: &Tensor, b: &Tensor) -> anyhow::Result<f32> {
    let a_f = a.to_dtype(DType::F32)?;
    let b_f = b.to_dtype(DType::F32)?;
    let dot = (&a_f * &b_f)?.sum_all()?.to_scalar::<f32>()?;
    let ma = a_f.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let mb = b_f.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    Ok(dot / (ma * mb))
}

fn main() -> anyhow::Result<()> {
    std::env::set_var("H3_ATTN", "attn8");
    std::env::set_var("H3_FORCE_GPU_ATT", "1");
    let dev = Device::new_cuda(0)?;

    let input_bundle = "/mnt/2tbnvS/tmp/stage1_input_768p_5s.edbin";
    let p8 = "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_w4a8_preswizzled.safetensors";
    let p4 = "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_50b_w4a4_preswizzled.safetensors";

    let mut mem = MemoryConfig::default();
    mem.dit_ram_blocks = 50;
    mem.offload = h3_rs::weights::Placement::Ram;

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
    let sigmas = vec![1.0, 0.982655942440033, 0.9146425724029541, 0.6747556924819946, 0.38522374629974365, 0.15905532240867615, 0.0];

    let mut gen = h3_rs::rng::TorchGen::new(42);
    let nv = gen.randn_tensor(&video_shape, &dev)?;
    let na = gen.randn_tensor(&[1, 32, 2, audio_t], &dev)?;
    let xv = (&nv * sigmas[0])?;
    let xa = (&na * sigmas[0])?;

    let blocks_to_capture = vec![0, 1, 2, 5, 10, 20, 30, 40, 49];

    println!("Loading W4A8 DiT...");
    let mut dit8 = Dit::load(p8, &dev, &mem, None)?;
    dit8.capture_blocks = blocks_to_capture.clone();
    let run8 = dit8.prepare(&cond, &video_shape, audio_t, sigmas.clone())?;

    println!("Forward W4A8...");
    let (ov8, _) = dit8.forward_inner(&run8, &xv, &xa, sigmas[0], sigmas[0])?;
    let cap8 = dit8.captured.take();
    drop(dit8);
    drop(run8);

    println!("Loading W4A4 DiT...");
    let mut dit4 = Dit::load(p4, &dev, &mem, None)?;
    dit4.capture_blocks = blocks_to_capture.clone();
    let run4 = dit4.prepare(&cond, &video_shape, audio_t, sigmas.clone())?;

    println!("Forward W4A4...");
    let (ov4, _) = dit4.forward_inner(&run4, &xv, &xa, sigmas[0], sigmas[0])?;
    let cap4 = dit4.captured.take();

    println!("=== BLOCK OUTPUT COMPARISON (W4A8 vs W4A4) ===");
    println!("cap8 indices: {:?}", cap8.iter().map(|(bi, _)| *bi).collect::<Vec<_>>());
    println!("cap4 indices: {:?}", cap4.iter().map(|(bi, _)| *bi).collect::<Vec<_>>());
    for (i, &(bi8, ref t8)) in cap8.iter().enumerate() {
        let (bi4, ref t4) = cap4[i];
        println!("i={}: bi8={}, bi4={}", i, bi8, bi4);
        let sim = cos_sim(t8, t4)?;
        let t8_f = t8.to_dtype(DType::F32)?;
        let t4_f = t4.to_dtype(DType::F32)?;
        let n8 = t8_f.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
        let n4 = t4_f.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
        println!("Block {:2}: cos_sim = {:.6}, norm(t8) = {:.4}, norm(t4) = {:.4}, ratio = {:.4}", bi8, sim, n8, n4, n4 / n8);
    }

    let final_sim = cos_sim(&ov8, &ov4)?;
    println!("Final velocity cos_sim: {:.6}", final_sim);

    Ok(())
}
