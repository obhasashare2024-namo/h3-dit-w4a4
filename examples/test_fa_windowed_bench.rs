use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    println!("=== Testing flash_attn_windowed on RTX 3070 Mobile (S=121,041, H=4, D=128) ===");

    let s = 121_041usize;
    let h = 4usize;
    let d = 128usize;
    let scale = 1.0 / (d as f32).sqrt();

    let q = Tensor::randn(0f32, 1.0, (1, s, h, d), &dev)?.to_dtype(DType::BF16)?;
    let k = Tensor::randn(0f32, 1.0, (1, s, h, d), &dev)?.to_dtype(DType::BF16)?;
    let v = Tensor::randn(0f32, 1.0, (1, s, h, d), &dev)?.to_dtype(DType::BF16)?;
    dev.synchronize()?;

    // Warmup
    let _ = candle_flash_attn::flash_attn(&q.narrow(1, 0, 1024)?, &k.narrow(1, 0, 1024)?, &v.narrow(1, 0, 1024)?, scale, false)?;
    dev.synchronize()?;

    // 1. Dense FA-2 Baseline
    println!("Running Dense FA-2...");
    let t0 = Instant::now();
    let out_dense = candle_flash_attn::flash_attn(&q, &k, &v, scale, false)?;
    dev.synchronize()?;
    let dense_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("Dense FA-2 Latency: {:.2} ms (Step: {:.1}s across 14 groups * 47 layers)",
        dense_ms, dense_ms * 14.0 * 47.0 / 1000.0);

    // 2. Windowed FA-2 with window_size = 4096 (left=4096, right=4096)
    for win in [8192, 4096, 2048, 1024] {
        println!("Running Windowed FA-2 (window_left={}, window_right={})...", win, win);
        let t1 = Instant::now();
        let out_win = candle_flash_attn::flash_attn_windowed(&q, &k, &v, scale, Some(win), Some(win))?;
        dev.synchronize()?;
        let win_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let speedup = dense_ms / win_ms;
        println!("  Window {} Latency: {:.2} ms (Speedup: {:.2}x, Step: {:.1}s)",
            win, win_ms, speedup, win_ms * 14.0 * 47.0 / 1000.0);
    }

    Ok(())
}
