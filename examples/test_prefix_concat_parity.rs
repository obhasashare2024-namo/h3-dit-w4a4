use candle_core::{Device, DType, Tensor};
use std::time::Instant;

fn cos_sim(a: &Tensor, b: &Tensor) -> anyhow::Result<f32> {
    let a_f = a.to_dtype(DType::F32)?;
    let b_f = b.to_dtype(DType::F32)?;
    let dot = (&a_f * &b_f)?.sum_all()?.to_scalar::<f32>()?;
    let ma = a_f.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let mb = b_f.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    Ok(dot / (ma * mb))
}

fn check_nan_inf(t: &Tensor) -> anyhow::Result<(bool, bool)> {
    let vec = t.to_dtype(DType::F32)?.narrow(1, 0, 1024.min(t.dim(1)?))?.flatten_all()?.to_vec1::<f32>()?;
    let mut has_nan = false;
    let mut has_inf = false;
    for &x in &vec {
        if x.is_nan() { has_nan = true; }
        if x.is_infinite() { has_inf = true; }
    }
    Ok((has_nan, has_inf))
}

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    let b = 1;
    let s = 121041;
    let p = 4761;
    let w = 8192;
    let h = 2;
    let d = 128;
    let scale = 1.0 / (d as f32).sqrt();

    println!("=== Parity & Benchmark: Prefix-Dense + Video-Window Attention ===");
    println!("S={}, P={}, W={}, H={}, D={}", s, p, w, h, d);

    let q = Tensor::randn(0f32, 1.0, (b, s, h, d), &dev)?.to_dtype(DType::F16)?;
    let k = Tensor::randn(0f32, 1.0, (b, s, h, d), &dev)?.to_dtype(DType::F16)?;
    let v = Tensor::randn(0f32, 1.0, (b, s, h, d), &dev)?.to_dtype(DType::F16)?;
    dev.synchronize()?;

    // 1. Dense Baseline
    println!("\n1. Running Dense FA-2 Baseline (121,041 x 121,041)...");
    let t0 = Instant::now();
    let out_dense = candle_flash_attn::flash_attn(&q, &k, &v, scale, false)?;
    dev.synchronize()?;
    let dense_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("   Dense Latency: {:.2} ms (Step: {:.1}s)", dense_ms, dense_ms * 672.0 / 1000.0);

    // 2. Pure Windowed
    println!("\n2. Running Pure Window 8192...");
    let t0 = Instant::now();
    let out_pure_win = candle_flash_attn::flash_attn_windowed(&q, &k, &v, scale, Some(w), Some(w))?;
    dev.synchronize()?;
    let pure_win_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let sim_pure = cos_sim(&out_dense, &out_pure_win)?;
    println!("   Pure Window Latency: {:.2} ms, Cosine vs Dense: {:.6}", pure_win_ms, sim_pure);

    // 3. Hybrid: Q_prefix Dense + Q_video Windowed
    println!("\n3. Running Hybrid: Q_prefix Dense (4761 x S) + Q_video Window (116280 x S, win=8192)...");
    let t0 = Instant::now();
    let q_p = q.narrow(1, 0, p)?;
    let q_v = q.narrow(1, p, s - p)?;
    let out_p = candle_flash_attn::flash_attn(&q_p, &k, &v, scale, false)?;
    let out_v = candle_flash_attn::flash_attn_windowed(&q_v, &k, &v, scale, Some(w), Some(w))?;
    let out_hybrid = Tensor::cat(&[&out_p, &out_v], 1)?;
    dev.synchronize()?;
    let hybrid_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (nan_h, inf_h) = check_nan_inf(&out_hybrid)?;
    let sim_hybrid = cos_sim(&out_dense, &out_hybrid)?;

    // Prefix-specific similarity
    let out_dense_p = out_dense.narrow(1, 0, p)?;
    let sim_prefix_only = cos_sim(&out_dense_p, &out_p)?;

    println!("   Hybrid Latency: {:.2} ms (Step: {:.1}s, Speedup: {:.2}x)",
        hybrid_ms, hybrid_ms * 672.0 / 1000.0, dense_ms / hybrid_ms);
    println!("   Prefix Tokens (0..4761) Cosine vs Dense: {:.6}", sim_prefix_only);
    println!("   Overall Sequence Cosine vs Dense:        {:.6}", sim_hybrid);
    println!("   NaN={}, Inf={}", nan_h, inf_h);

    Ok(())
}
