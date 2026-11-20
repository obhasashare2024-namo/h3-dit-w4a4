use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    let s = 121_041usize;
    let d = 128usize;
    let total_heads = 56usize;

    println!("=== FlashAttention Scaling Benchmark on Ampere sm_86 ===");
    println!("Config: S={} (1080p 8s), HeadDim={}", s, d);

    let head_configs = [2, 4, 7, 8, 14];

    for &h in &head_configs {
        if total_heads % h != 0 { continue; }
        let groups = total_heads / h;
        let q = Tensor::randn(0f32, 1.0, (1, s, h, d), &dev)?.to_dtype(DType::F16)?;
        let k = Tensor::randn(0f32, 1.0, (1, s, h, d), &dev)?.to_dtype(DType::F16)?;
        let v = Tensor::randn(0f32, 1.0, (1, s, h, d), &dev)?.to_dtype(DType::F16)?;

        // Warmup
        dev.synchronize()?;
        let _ = h3_rs::ops::attention(&q, &k, &v, false)?;
        dev.synchronize()?;

        let mut iters = Vec::new();
        for _ in 0..3 {
            dev.synchronize()?;
            let t0 = Instant::now();
            let _ = h3_rs::ops::attention(&q, &k, &v, false)?;
            dev.synchronize()?;
            iters.push(t0.elapsed().as_secs_f64());
        }
        let avg_group_ms = iters.iter().sum::<f64>() / iters.len() as f64 * 1000.0;
        let block_ms = avg_group_ms * groups as f64;
        let layer47_s = block_ms * 47.0 / 1000.0;
        println!("Heads/Grp={:2}, Groups={:2} | Group Latency: {:6.2} ms | 1 Block: {:6.2} ms | 47 Layers: {:6.2} s",
            h, groups, avg_group_ms, block_ms, layer47_s);
    }
    Ok(())
}
