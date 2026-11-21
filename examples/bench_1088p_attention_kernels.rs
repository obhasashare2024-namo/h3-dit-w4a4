use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    let s = 121_041usize;
    let h = 2usize;
    let d = 128usize;
    let w = h * d;
    let groups = 14usize;
    let layers = 48usize;
    let total_group_calls_per_step = groups * layers; // 672 calls

    println!("====================================================================");
    println!("  NATIVE ACCELERATION TEST 3: 1088P (S=121,041) Attention Benchmark ");
    println!("  Comparing: FlashAttention-2 (Dense FP16) vs Attn8 (SageAttn INT8) ");
    println!("====================================================================");

    // Synthetic Q, K, V for one group [S, H, D]
    let q = Tensor::randn(0f32, 1.0, (s, h, d), &dev)?.to_dtype(DType::BF16)?;
    let k_offset = Tensor::randn(0f32, 1.0, (1, h, d), &dev)?.to_dtype(DType::BF16)?.broadcast_as((s, h, d))?;
    let k = (Tensor::randn(0f32, 1.0, (s, h, d), &dev)?.to_dtype(DType::BF16)? + k_offset)?;
    let vt = Tensor::randn(0f32, 1.0, (s, w), &dev)?.to_dtype(DType::BF16)?;
    let v_flash = vt.reshape((s, h, d))?;

    let q_4d = q.unsqueeze(0)?;
    let k_4d = k.unsqueeze(0)?;
    let v_4d = v_flash.unsqueeze(0)?;

    // 1. FlashAttention-2 (FP16 Dense)
    println!("--- 1. Testing FlashAttention-2 (FP16 Dense) ---");
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
    let fa_step_s = (fa_avg_ms * total_group_calls_per_step as f64) / 1000.0;
    println!("  FlashAttention-2 average: {:.2} ms per group", fa_avg_ms);
    println!("  FlashAttention-2 step:    {:.2} s per step (672 group calls)", fa_step_s);

    // 2. Attn8 (SageAttention INT8-QK sm_86)
    println!("\n--- 2. Testing Attn8 (SageAttention INT8-QK sm_86) ---");
    assert!(h3_rs::attn8::available(&q), "attn8 must be available on CUDA sm_86!");

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
    let int8_step_s = (int8_avg_ms * total_group_calls_per_step as f64) / 1000.0;
    println!("  Attn8 average:            {:.2} ms per group", int8_avg_ms);
    println!("  Attn8 step:               {:.2} s per step (672 group calls)", int8_step_s);

    // 3. Comparison
    println!("\n--- 3. Kernel Comparison & Extrapolation ---");
    if fa_avg_ms < int8_avg_ms {
        let speedup = int8_avg_ms / fa_avg_ms;
        println!("  FlashAttention-2 is faster by {:.2}x ({:.2} ms vs {:.2} ms)", speedup, fa_avg_ms, int8_avg_ms);
        println!("  Net Attention Savings: {:.2} s per step (8 steps saves {:.1} minutes!)", int8_step_s - fa_step_s, (int8_step_s - fa_step_s) * 8.0 / 60.0);
    } else {
        let speedup = fa_avg_ms / int8_avg_ms;
        println!("  Attn8 is faster by {:.2}x ({:.2} ms vs {:.2} ms)", speedup, int8_avg_ms, fa_avg_ms);
        println!("  Net Attention Savings: {:.2} s per step (8 steps saves {:.1} minutes!)", fa_step_s - int8_step_s, (fa_step_s - int8_step_s) * 8.0 / 60.0);
    }

    Ok(())
}
