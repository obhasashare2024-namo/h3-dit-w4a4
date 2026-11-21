use std::fs::File;
use std::io::Read;
use std::time::Instant;
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

fn norm(t: &Tensor) -> anyhow::Result<f32> {
    Ok(t.to_dtype(DType::F32)?.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt())
}

fn main() -> anyhow::Result<()> {
    std::env::set_var("H3_ATTN", "flash");
    std::env::set_var("H3_ATTN_GROUPS", "14");
    std::env::set_var("H3_STREAMING", "1");
    std::env::set_var("H3_FORCE_GPU_ATT", "0");
    std::env::set_var("H3_ZERO_OFFLOAD", "1");
    println!("=== 1-STEP 1088P 8S BENCHMARK: DENSE FA-2 vs WINDOWED FA-2 ===");

    let dev = Device::new_cuda(0)?;
    let input_bundle = "/mnt/2tbnvS/tmp/stage1_input_1080p_8s.edbin";
    let p4 = "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.safetensors";

    let mut mem = MemoryConfig::default();
    mem.dit_vram_blocks = 0;
    mem.dit_ram_blocks = 50;
    mem.dit_disk_blocks = 0;
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

    println!("Input video shape: {:?}, tokens S = {}", video_shape, xv.dim(0)?);
    println!(">>> Loading W4A4 DiT...");
    let dit = Dit::load(p4, &dev, &mem, None)?;
    let run = dit.prepare(&cond, &video_shape, audio_t, sigmas.clone())?;

    // 1. Dense Baseline (H3_ATTN_WINDOW unset)
    std::env::remove_var("H3_ATTN_WINDOW");
    println!("\n[1/2] Running Step 0 with Dense FA-2 (Baseline)...");
    let t0 = Instant::now();
    let (ov_dense, _) = dit.forward_inner(&run, &xv, &xa, sigmas[0], sigmas[0])?;
    dev.synchronize()?;
    let t_dense = t0.elapsed().as_secs_f32();
    let n_dense = norm(&ov_dense)?;
    println!("  Dense Step 0 completed in {:.2}s, norm = {:.4}", t_dense, n_dense);

    // 2. Windowed Attention (H3_ATTN_WINDOW=8192)
    std::env::set_var("H3_ATTN_WINDOW", "8192");
    println!("\n[2/2] Running Step 0 with Windowed FA-2 (H3_ATTN_WINDOW=8192)...");
    let t1 = Instant::now();
    let (ov_win, _) = dit.forward_inner(&run, &xv, &xa, sigmas[0], sigmas[0])?;
    dev.synchronize()?;
    let t_win = t1.elapsed().as_secs_f32();
    let n_win = norm(&ov_win)?;
    let sim = cos_sim(&ov_dense, &ov_win)?;
    let speedup = t_dense / t_win;
    println!("  Window 8192 Step 0 completed in {:.2}s (Speedup: {:.2}x), norm = {:.4}", t_win, speedup, n_win);
    println!("  Cosine Similarity vs Dense: {:.6}, Norm Ratio: {:.4}", sim, n_win / n_dense);

    println!("\n========================================================");
    println!("SUMMARY:");
    println!("  Dense Step 0     : {:.2}s", t_dense);
    println!("  Window 8192 Step 0: {:.2}s (Net Saving: {:.2}s / step)", t_win, t_dense - t_win);
    println!("  Full 6-step est  : {:.1}m (Windowed) vs {:.1}m (Dense)", (t_win * 6.0) / 60.0, (t_dense * 6.0) / 60.0);

    Ok(())
}
