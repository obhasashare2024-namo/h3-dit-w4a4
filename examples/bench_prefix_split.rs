use std::time::Instant;
use candle_core::{Device, DType, Tensor};

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    let b = 1;
    let s = 121041;
    let p = 4761;
    let v_len = s - p; // 116280
    let h = 2;
    let d = 128;
    let scale = 1.0 / (d as f32).sqrt();

    println!("=== FULL BIDIRECTIONAL PREFIX-WINDOW ATTENTION BENCHMARK ===");
    println!("Sequence S={}, Prefix P={}, Video V={}", s, p, v_len);

    let q_p = Tensor::randn(0f32, 1f32, (b, p, h, d), &dev)?.to_dtype(DType::F16)?;
    let q_v = Tensor::randn(0f32, 1f32, (b, v_len, h, d), &dev)?.to_dtype(DType::F16)?;
    let k_p = Tensor::randn(0f32, 1f32, (b, p, h, d), &dev)?.to_dtype(DType::F16)?;
    let v_p = Tensor::randn(0f32, 1f32, (b, p, h, d), &dev)?.to_dtype(DType::F16)?;
    let k_all = Tensor::randn(0f32, 1f32, (b, s, h, d), &dev)?.to_dtype(DType::F16)?;
    let v_all = Tensor::randn(0f32, 1f32, (b, s, h, d), &dev)?.to_dtype(DType::F16)?;
    let k_v = Tensor::randn(0f32, 1f32, (b, v_len, h, d), &dev)?.to_dtype(DType::F16)?;
    let v_v = Tensor::randn(0f32, 1f32, (b, v_len, h, d), &dev)?.to_dtype(DType::F16)?;

    // 1. Q_prefix (4761) -> All K, V (121041) Dense
    let iters = 10;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = candle_flash_attn::flash_attn(&q_p, &k_all, &v_all, scale, false)?;
    }
    dev.synchronize()?;
    let t_qp = t0.elapsed().as_secs_f64() / iters as f64;
    println!("[1] Q_prefix (4761) -> K_all (121041) Dense:       {:.2} ms (Step: {:.2}s)", t_qp * 1000.0, t_qp * 672.0);

    // 2. Q_video (116280) -> K_prefix (4761) Cross Attention
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = candle_flash_attn::flash_attn(&q_v, &k_p, &v_p, scale, false)?;
    }
    dev.synchronize()?;
    let t_qv_kp = t0.elapsed().as_secs_f64() / iters as f64;
    println!("[2] Q_video (116280) -> K_prefix (4761) Cross:     {:.2} ms (Step: {:.2}s)", t_qv_kp * 1000.0, t_qv_kp * 672.0);

    // 3. Q_video (116280) -> K_video (116280) Window 8192
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = candle_flash_attn::flash_attn_windowed(&q_v, &k_v, &v_v, scale, Some(8192), Some(8192))?;
    }
    dev.synchronize()?;
    let t_win = t0.elapsed().as_secs_f64() / iters as f64;
    println!("[3] Q_video (116280) -> K_video (116280) Win8192:   {:.2} ms (Step: {:.2}s)", t_win * 1000.0, t_win * 672.0);

    let total_ms = (t_qp + t_qv_kp + t_win) * 1000.0;
    let total_step = (t_qp + t_qv_kp + t_win) * 672.0;
    println!("----------------------------------------------------------------");
    println!("TOTAL BIDIRECTIONAL PREFIX + WINDOW ATTENTION:");
    println!("Single group: {:.2} ms | Whole Model (672 groups): {:.2} seconds / step", total_ms, total_step);
    println!("Dense FA-2 Baseline: 570 ms / group (383 seconds / step)");
    println!("Net Speedup: {:.2}x faster Attention!", 383.0 / total_step);
    Ok(())
}
