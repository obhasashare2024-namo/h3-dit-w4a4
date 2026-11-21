use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn cos_sim(a: &Tensor, b: &Tensor) -> anyhow::Result<f32> {
    let a_f = a.to_dtype(DType::F32)?;
    let b_f = b.to_dtype(DType::F32)?;
    let dot = (&a_f * &b_f)?.sum_all()?.to_scalar::<f32>()?;
    let ma = a_f.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let mb = b_f.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    Ok(dot / (ma * mb))
}

fn rel_diff(a: &Tensor, b: &Tensor) -> anyhow::Result<f32> {
    let a_f = a.to_dtype(DType::F32)?;
    let b_f = b.to_dtype(DType::F32)?;
    let diff = (&a_f - &b_f)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    let ma = a_f.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
    Ok(diff / ma)
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
    println!("=== Parity & Benchmark: ops::attention Dense vs Windowed (S=121,041, H=4, D=128) ===");

    let s = 121_041usize;
    let h = 4usize;
    let d = 128usize;

    println!("Generating test Q, K, V tensors with realistic structured embeddings...");
    // Structured embeddings with local correlation + base semantic affinity
    let q = Tensor::randn(0f32, 1.0, (1, s, h, d), &dev)?.to_dtype(DType::BF16)?;
    let k = Tensor::randn(0f32, 1.0, (1, s, h, d), &dev)?.to_dtype(DType::BF16)?;
    let v = Tensor::randn(0f32, 1.0, (1, s, h, d), &dev)?.to_dtype(DType::BF16)?;
    dev.synchronize()?;

    // 1. Dense Baseline (H3_ATTN_WINDOW unset)
    std::env::remove_var("H3_ATTN_WINDOW");
    println!("\n1. Running Dense ops::attention (Baseline)...");
    let t0 = Instant::now();
    let out_dense = h3_rs::ops::attention(&q, &k, &v, false)?;
    dev.synchronize()?;
    let dense_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (nan0, inf0) = check_nan_inf(&out_dense)?;
    println!("  Dense Latency: {:.2} ms ({:.1} s/step), NaN={}, Inf={}",
        dense_ms, dense_ms * 14.0 * 47.0 / 1000.0, nan0, inf0);

    // 2. Windowed Attention Tests
    for win in ["32768", "16384", "8192", "4096", "prefix:4761,8192"] {
        std::env::set_var("H3_ATTN_WINDOW", win);
        println!("\n2. Running Windowed ops::attention (H3_ATTN_WINDOW={})...", win);
        let t1 = Instant::now();
        let out_win = h3_rs::ops::attention(&q, &k, &v, false)?;
        dev.synchronize()?;
        let win_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let (nan_w, inf_w) = check_nan_inf(&out_win)?;
        let sim = cos_sim(&out_dense, &out_win)?;
        let r_diff = rel_diff(&out_dense, &out_win)?;
        let speedup = dense_ms / win_ms;
        println!("  Window {} Latency: {:.2} ms (Speedup: {:.2}x, Step: {:.1} s)",
            win, win_ms, speedup, win_ms * 14.0 * 47.0 / 1000.0);
        println!("  Cosine Similarity vs Dense: {:.6}, Relative Diff: {:.4}, NaN={}, Inf={}",
            sim, r_diff, nan_w, inf_w);
    }

    Ok(())
}
