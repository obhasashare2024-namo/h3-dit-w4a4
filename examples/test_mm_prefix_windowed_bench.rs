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

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    println!("=== Testing flash_attn_varlen_paged_windowed with mm_prefix_ranges on RTX 3070 Mobile ===");

    let s = 121_041usize;
    let h = 4usize;
    let d = 128usize;
    let scale = 1.0 / (d as f32).sqrt();
    let page_block_size = 128usize;
    let num_blocks = s.div_ceil(page_block_size);
    let pad_s = num_blocks * page_block_size;
    let pad_len = pad_s - s;

    println!("Sequence metrics: S={}, H={}, D={}, BlockSize={}, NumBlocks={}, Pad={}",
        s, h, d, page_block_size, num_blocks, pad_len);

    let q = Tensor::randn(0f32, 1.0, (s, h, d), &dev)?.to_dtype(DType::BF16)?;
    let k = Tensor::randn(0f32, 1.0, (s, h, d), &dev)?.to_dtype(DType::BF16)?;
    let v = Tensor::randn(0f32, 1.0, (s, h, d), &dev)?.to_dtype(DType::BF16)?;

    // Paged K and V
    let pad_k = Tensor::zeros((pad_len, h, d), DType::BF16, &dev)?;
    let pad_v = Tensor::zeros((pad_len, h, d), DType::BF16, &dev)?;
    let k_paged = Tensor::cat(&[&k, &pad_k], 0)?.reshape((num_blocks, page_block_size, h, d))?;
    let v_paged = Tensor::cat(&[&v, &pad_v], 0)?.reshape((num_blocks, page_block_size, h, d))?;

    let seqlens = Tensor::new(&[0u32, s as u32], &dev)?;
    let block_table = Tensor::new((0..num_blocks as u32).collect::<Vec<_>>(), &dev)?.reshape((1, num_blocks))?;

    // Dense FA-2 Baseline
    println!("Running Dense FA-2 Baseline...");
    let q4 = q.unsqueeze(0)?;
    let k4 = k.unsqueeze(0)?;
    let v4 = v.unsqueeze(0)?;
    let t0 = Instant::now();
    let out_dense = candle_flash_attn::flash_attn(&q4, &k4, &v4, scale, false)?;
    dev.synchronize()?;
    let dense_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Dense FA-2 Latency: {:.2} ms (Step: {:.1}s)", dense_ms, dense_ms * 14.0 * 47.0 / 1000.0);

    // Test with mm_prefix_ranges: [0, 4761] (Extra text/cond/audio tokens)
    let extra_tokens = 4_761i32;
    let mm_prefix = Tensor::new(&[0i32, extra_tokens], &dev)?.reshape((1, 1, 2))?;

    for win in [8192, 4096, 2048] {
        println!("\nRunning Paged Windowed with MM Prefix (window={}, prefix=[0, {}])...", win, extra_tokens);
        let t1 = Instant::now();
        let out_paged = candle_flash_attn::flash_attn_varlen_paged_windowed(
            &q,
            &k_paged,
            &v_paged,
            &seqlens,
            &seqlens,
            &block_table,
            Some(&mm_prefix),
            s,
            s,
            scale,
            Some(win),
            Some(win),
            page_block_size,
            None,
        )?;
        dev.synchronize()?;
        let paged_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let speedup = dense_ms / paged_ms;
        let sim = cos_sim(&out_dense.squeeze(0)?, &out_paged)?;
        let r_diff = rel_diff(&out_dense.squeeze(0)?, &out_paged)?;
        println!("  Window {} + MM-Prefix Latency: {:.2} ms (Speedup: {:.2}x, Step: {:.1}s)",
            win, paged_ms, speedup, paged_ms * 14.0 * 47.0 / 1000.0);
        println!("  Cosine Similarity vs Dense: {:.6}, Relative Diff: {:.4}", sim, r_diff);
    }

    Ok(())
}
