use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn bench_shape(dev: &Device, name: &str, s: usize, h: usize, d: usize) -> anyhow::Result<()> {
    let w = h * d;
    println!("\n=======================================================");
    println!("Testing Shape [{}]: S={}, H={}, D={}, W={}", name, s, h, d, w);
    println!("=======================================================");

    let q = Tensor::randn(0f32, 1.0, (s, h, d), dev)?.to_dtype(DType::BF16)?;
    let k_offset = Tensor::randn(0f32, 2.0, (1, h, d), dev)?.to_dtype(DType::BF16)?.broadcast_as((s, h, d))?;
    let k = (Tensor::randn(0f32, 1.0, (s, h, d), dev)?.to_dtype(DType::BF16)? + k_offset)?;
    let vt = Tensor::randn(0f32, 1.0, (s, w), dev)?.to_dtype(DType::BF16)?;

    let v_flash = vt.reshape((s, h, d))?;
    let q_4d = q.unsqueeze(0)?;
    let k_4d = k.unsqueeze(0)?;
    let v_4d = v_flash.unsqueeze(0)?;

    // 1. FlashAttention (FP16)
    dev.synchronize()?;
    let _ = h3_rs::ops::attention(&q_4d, &k_4d, &v_4d, false)?;
    dev.synchronize()?;

    let mut fa_times = Vec::new();
    for _ in 0..5 {
        dev.synchronize()?;
        let t0 = Instant::now();
        let _ = h3_rs::ops::attention(&q_4d, &k_4d, &v_4d, false)?;
        dev.synchronize()?;
        fa_times.push(t0.elapsed().as_secs_f64());
    }
    let fa_avg_ms = fa_times.iter().sum::<f64>() / fa_times.len() as f64 * 1000.0;
    println!("FlashAttention FP16 average: {:.2} ms per group", fa_avg_ms);

    // 2. Attn8 (INT8-QK + Lazy Rescaling)
    dev.synchronize()?;
    let _ = h3_rs::attn8::attention_qkv(&q, &k, &vt, 0)?;
    dev.synchronize()?;

    let mut int8_times = Vec::new();
    for _ in 0..5 {
        dev.synchronize()?;
        let t0 = Instant::now();
        let _ = h3_rs::attn8::attention_qkv(&q, &k, &vt, 0)?;
        dev.synchronize()?;
        int8_times.push(t0.elapsed().as_secs_f64());
    }
    let int8_avg_ms = int8_times.iter().sum::<f64>() / int8_times.len() as f64 * 1000.0;
    println!("Attn8 INT8-QK average:      {:.2} ms per group", int8_avg_ms);

    let speedup = fa_avg_ms / int8_avg_ms;
    println!("===> Ratio (FA/Attn8): {:.2}x ({:.2} ms vs {:.2} ms)", speedup, fa_avg_ms, int8_avg_ms);

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    bench_shape(&dev, "1080p 8s Shape (H=14)", 121_041, 14, 128)?;
    bench_shape(&dev, "1080p 12s Shape (H=2)", 182_581, 2, 128)?;
    bench_shape(&dev, "1080p 12s with H=4 Test", 182_581, 4, 128)?;
    bench_shape(&dev, "1080p 12s with H=7 Test", 182_581, 7, 128)?;
    Ok(())
}
