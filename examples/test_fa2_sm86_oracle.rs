use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn cosine_similarity(a: &Tensor, b: &Tensor) -> candle_core::Result<f32> {
    let a_f32 = a.to_dtype(DType::F32)?;
    let b_f32 = b.to_dtype(DType::F32)?;
    let dot = (&a_f32 * &b_f32)?.sum_all()?.to_scalar::<f32>()?;
    let norm_a = a_f32.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let norm_b = b_f32.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    Ok(dot / (norm_a * norm_b))
}

fn rel_l2_diff(a: &Tensor, b: &Tensor) -> candle_core::Result<f32> {
    let a_f32 = a.to_dtype(DType::F32)?;
    let b_f32 = b.to_dtype(DType::F32)?;
    let diff = (&a_f32 - &b_f32)?.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let norm_a = a_f32.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    Ok(diff / norm_a)
}

fn check_nan_inf(t: &Tensor) -> candle_core::Result<(bool, bool)> {
    let vec = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut has_nan = false;
    let mut has_inf = false;
    for &x in &vec {
        if x.is_nan() { has_nan = true; }
        if x.is_infinite() { has_inf = true; }
        if has_nan && has_inf { break; }
    }
    Ok((has_nan, has_inf))
}

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    println!("=================================================================");
    println!("=== DEF-077: FlashAttention-2 sm_86 Precision Oracle Parity ===");
    println!("=================================================================");

    let d = 128usize;

    // Test cases: (name, S, H, dtype)
    let test_cases = [
        ("S=1024 H=4 FP16", 1024, 4, DType::F16),
        ("S=1024 H=4 BF16", 1024, 4, DType::BF16),
        ("S=8439 H=4 FP16", 8439, 4, DType::F16),
        ("S=8439 H=4 BF16", 8439, 4, DType::BF16),
    ];

    for &(name, s, h, dtype) in &test_cases {
        println!("\n--- Case: {} ---", name);
        let q = Tensor::randn(0f32, 1.0, (1, s, h, d), &dev)?.to_dtype(dtype)?;
        let k = Tensor::randn(0f32, 1.0, (1, s, h, d), &dev)?.to_dtype(dtype)?;
        let v = Tensor::randn(0f32, 1.0, (1, s, h, d), &dev)?.to_dtype(dtype)?;

        let scale = 1.0 / (d as f32).sqrt();

        // 1. Oracle: naive chunked attention_fallback in FP32
        let t0 = Instant::now();
        let oracle_out = h3_rs::ops::attention_fallback(&q, &k, &v, scale, false)?;
        dev.synchronize()?;
        let oracle_time = t0.elapsed().as_secs_f64() * 1000.0;

        // 2. FlashAttention-2 (via h3_rs::ops::attention)
        let t1 = Instant::now();
        let fa_out = h3_rs::ops::attention(&q, &k, &v, false)?;
        dev.synchronize()?;
        let fa_time = t1.elapsed().as_secs_f64() * 1000.0;

        // 3. Parity metrics
        let cos = cosine_similarity(&oracle_out, &fa_out)?;
        let rel_diff = rel_l2_diff(&oracle_out, &fa_out)?;
        let (has_nan, has_inf) = check_nan_inf(&fa_out)?;

        println!("  Oracle Time: {:8.2} ms | FA-2 Time: {:8.2} ms (Speedup: {:.2}x)",
            oracle_time, fa_time, oracle_time / fa_time);
        println!("  Cosine Similarity: {:.6} (Gate > 0.999)", cos);
        println!("  Relative L2 Diff : {:.6} (Gate < 0.05)", rel_diff);
        println!("  Has NaN: {} | Has Inf: {}", has_nan, has_inf);

        assert!(cos > 0.999, "Cosine similarity failed gate!");
        assert!(rel_diff < 0.05, "Relative L2 diff failed gate!");
        assert!(!has_nan && !has_inf, "Output has NaN or Inf!");
        println!("  ===> STATUS: [PASS]");
    }

    println!("\n--- Case: S=121041 H=4 BF16 (1080P 8s Full Scale Single Group) ---");
    let s = 121041usize;
    let h = 4usize;
    let q = Tensor::randn(0f32, 1.0, (1, s, h, d), &dev)?.to_dtype(DType::BF16)?;
    let k = Tensor::randn(0f32, 1.0, (1, s, h, d), &dev)?.to_dtype(DType::BF16)?;
    let v = Tensor::randn(0f32, 1.0, (1, s, h, d), &dev)?.to_dtype(DType::BF16)?;

    dev.synchronize()?;
    let t0 = Instant::now();
    let fa_out = h3_rs::ops::attention(&q, &k, &v, false)?;
    dev.synchronize()?;
    let fa_time = t0.elapsed().as_secs_f64() * 1000.0;

    let (has_nan, has_inf) = check_nan_inf(&fa_out)?;
    let mean = fa_out.to_dtype(DType::F32)?.mean_all()?.to_scalar::<f32>()?;
    let var = (fa_out.to_dtype(DType::F32)? - mean as f64)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let std = var.sqrt();

    println!("  FA-2 S=121,041 Time: {:8.2} ms", fa_time);
    println!("  FA-2 Output Stats : Mean = {:.6}, Std = {:.6}", mean, std);
    println!("  Has NaN: {} | Has Inf: {}", has_nan, has_inf);
    assert!(!has_nan && !has_inf, "Output has NaN or Inf at S=121,041!");
    println!("  ===> STATUS: [PASS]");

    println!("\n=================================================================");
    println!("=== ALL ORACLE PARITY GATES PASSED (DEF-077 VERIFIED) ===");
    println!("=================================================================");

    Ok(())
}
