use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    let s = 121_041usize;
    let d = 128usize;
    let total_heads = 56usize;

    println!("=== FlashAttention vs Attn8 Scaling Benchmark on sm_86 ===");
    println!("Config: S={} (1080p 8s), HeadDim={}", s, d);

    let head_configs = [2, 4, 7, 8, 14];

    for &h in &head_configs {
        if total_heads % h != 0 { continue; }
        let groups = total_heads / h;
        let w = h * d;

        let q = Tensor::randn(0f32, 1.0, (s, h, d), &dev)?.to_dtype(DType::BF16)?;
        let k_offset = Tensor::randn(0f32, 2.0, (1, h, d), &dev)?.to_dtype(DType::BF16)?.broadcast_as((s, h, d))?;
        let k = (Tensor::randn(0f32, 1.0, (s, h, d), &dev)?.to_dtype(DType::BF16)? + k_offset)?;
        let vt = Tensor::randn(0f32, 1.0, (s, w), &dev)?.to_dtype(DType::BF16)?;

        // 1. FlashAttention
        let v_fa = vt.reshape((s, h, d))?;
        let q_4d = q.unsqueeze(0)?;
        let k_4d = k.unsqueeze(0)?;
        let v_4d = v_fa.unsqueeze(0)?;

        dev.synchronize()?;
        let _ = h3_rs::ops::attention(&q_4d, &k_4d, &v_4d, false)?;
        dev.synchronize()?;

        let t0 = Instant::now();
        for _ in 0..3 {
            let _ = h3_rs::ops::attention(&q_4d, &k_4d, &v_4d, false)?;
        }
        dev.synchronize()?;
        let fa_ms = t0.elapsed().as_secs_f64() / 3.0 * 1000.0;

        // 2. Attn8
        dev.synchronize()?;
        let _ = h3_rs::attn8::attention_qkv(&q, &k, &vt, 0)?;
        dev.synchronize()?;

        let t1 = Instant::now();
        for _ in 0..3 {
            let _ = h3_rs::attn8::attention_qkv(&q, &k, &vt, 0)?;
        }
        dev.synchronize()?;
        let attn8_ms = t1.elapsed().as_secs_f64() / 3.0 * 1000.0;

        let speedup = fa_ms / attn8_ms;
        println!("H={:2}, G={:2} | FA: {:6.2} ms | Attn8: {:6.2} ms | Speedup: {:.2}x | 47L FA: {:5.1}s, Attn8: {:5.1}s",
            h, groups, fa_ms, attn8_ms, speedup, fa_ms * groups as f64 * 47.0 / 1000.0, attn8_ms * groups as f64 * 47.0 / 1000.0);
    }
    Ok(())
}
