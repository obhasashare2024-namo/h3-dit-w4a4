use candle_core::{DType, Device, Result, Tensor};
use std::time::Instant;

fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a_f32 = a.to_dtype(DType::F32)?;
    let b_f32 = b.to_dtype(DType::F32)?;
    let dot = (&a_f32 * &b_f32)?.sum_all()?.to_scalar::<f32>()?;
    let norm_a = a_f32.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let norm_b = b_f32.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }
    Ok(dot / (norm_a * norm_b))
}

fn rel_l2_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a_f32 = a.to_dtype(DType::F32)?;
    let b_f32 = b.to_dtype(DType::F32)?;
    let diff = (&a_f32 - &b_f32)?.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let norm_a = a_f32.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    if norm_a == 0.0 {
        return Ok(0.0);
    }
    Ok(diff / norm_a)
}

fn check_nan_inf(t: &Tensor) -> Result<(bool, bool)> {
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
    println!("=== DEF-083: Phase 7B 3D Block-Sparse Attention Benchmark     ===");
    println!("=================================================================");

    let s = 121_041usize;
    let h = 4usize; // 1 group of 4 heads (56 / 14 = 4)
    let d = 128usize;
    let extra_tokens = 4_761usize;
    let video_tokens = 116_280usize;
    let tile_size = 64usize;
    let full_tiles = video_tokens / tile_size; // 1816
    let rem_tokens = video_tokens % tile_size; // 56
    let num_tiles = if rem_tokens > 0 { full_tiles + 1 } else { full_tiles }; // 1817

    println!("Sequence Metrics:");
    println!("  Total S       : {} (Extra: {}, Video: {})", s, extra_tokens, video_tokens);
    println!("  Heads Group   : H={} (D={})", h, d);
    println!("  Tile Geometry : TileSize={}, FullTiles={}, RemTiles={}, TotalTiles={}",
        tile_size, full_tiles, if rem_tokens > 0 { 1 } else { 0 }, num_tiles);

    // 1. Allocate Q, K, V for one group [1, S, H, D]
    println!("\nAllocating BF16 Q, K, V tensors on GPU...");
    let t_alloc = Instant::now();
    let q = Tensor::randn(0f32, 1.0, (1, s, h, d), &dev)?.to_dtype(DType::BF16)?;
    let k = Tensor::randn(0f32, 1.0, (1, s, h, d), &dev)?.to_dtype(DType::BF16)?;
    let v = Tensor::randn(0f32, 1.0, (1, s, h, d), &dev)?.to_dtype(DType::BF16)?;
    dev.synchronize()?;
    println!("Allocated in {:.2} ms", t_alloc.elapsed().as_secs_f64() * 1000.0);

    let scale = 1.0 / (d as f32).sqrt();

    // 2. Measure Dense FA-2 Baseline
    println!("\nMeasuring Dense FlashAttention-2 Baseline [1, {}, {}, {}]...", s, h, d);
    let t0 = Instant::now();
    let out_dense = candle_flash_attn::flash_attn(&q, &k, &v, scale, false)?;
    dev.synchronize()?;
    let dense_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("Dense FA-2 Latency: {:.2} ms ({:.2} s/layer across 14 groups, {:.1} s/step for 47 layers)",
        dense_ms, dense_ms * 14.0 / 1000.0, dense_ms * 14.0 * 47.0 / 1000.0);

    let (nan_d, inf_d) = check_nan_inf(&out_dense.narrow(1, 0, 1024)?)?;
    println!("Dense Output Sanity: NaN={}, Inf={}", nan_d, inf_d);
    assert!(!nan_d && !inf_d, "Dense output has NaN/Inf!");

    // 3. Pooled Global Representation Construction
    println!("\nBuilding 64-Tile Pooled Global representations...");
    let t_pool = Instant::now();
    // video_k is [1, video_tokens, h, d] -> permute to [video_tokens, h, d]
    let vk = k.narrow(1, extra_tokens, video_tokens)?.squeeze(0)?;
    let vv = v.narrow(1, extra_tokens, video_tokens)?.squeeze(0)?;

    let vk_full = vk.narrow(0, 0, full_tiles * tile_size)?.reshape((full_tiles, tile_size, h, d))?;
    let vv_full = vv.narrow(0, 0, full_tiles * tile_size)?.reshape((full_tiles, tile_size, h, d))?;
    let pooled_k_main = vk_full.mean(1)?; // [full_tiles, h, d]
    let pooled_v_main = vv_full.mean(1)?;

    let (pooled_k, pooled_v) = if rem_tokens > 0 {
        let rem_k = vk.narrow(0, full_tiles * tile_size, rem_tokens)?.mean(0)?.reshape((1, h, d))?;
        let rem_v = vv.narrow(0, full_tiles * tile_size, rem_tokens)?.mean(0)?.reshape((1, h, d))?;
        (Tensor::cat(&[&pooled_k_main, &rem_k], 0)?, Tensor::cat(&[&pooled_v_main, &rem_v], 0)?)
    } else {
        (pooled_k_main, pooled_v_main)
    };

    let ek = k.narrow(1, 0, extra_tokens)?.squeeze(0)?; // [extra_tokens, h, d]
    let ev = v.narrow(1, 0, extra_tokens)?.squeeze(0)?;

    let anchors_k = Tensor::cat(&[&ek, &pooled_k], 0)?.unsqueeze(0)?; // [1, extra + num_tiles, h, d]
    let anchors_v = Tensor::cat(&[&ev, &pooled_v], 0)?.unsqueeze(0)?;

    dev.synchronize()?;
    let pool_ms = t_pool.elapsed().as_secs_f64() * 1000.0;
    println!("Global Anchor Bank built in {:.2} ms (Anchors: {} tokens)",
        pool_ms, extra_tokens + num_tiles);

    // 4. Measure Global Anchor Branch Latency via FA-2
    // S_anchor = 4,761 + 1,817 = 6,578 tokens!
    println!("\nMeasuring Global Anchor Branch Latency (S_q={}, S_kv={})...", s, extra_tokens + num_tiles);
    let t_anchor = Instant::now();
    // Compute attention of all queries against global anchors
    let out_global = candle_flash_attn::flash_attn(&q, &anchors_k, &anchors_v, scale, false)?;
    dev.synchronize()?;
    let anchor_ms = t_anchor.elapsed().as_secs_f64() * 1000.0;
    println!("Global Anchor Branch Latency: {:.2} ms", anchor_ms);

    // 5. Measure 3D Local Window Branch Latency via FA-2
    // Local window: S_local = 4096 tokens per query block
    // Processing query chunks of size 4096 against their local 4096 window
    let chunk_size = 4096usize;
    let num_chunks = video_tokens.div_ceil(chunk_size);
    println!("\nMeasuring Local Window Branch Latency ({} chunks of size {})...", num_chunks, chunk_size);
    let t_local = Instant::now();

    for ci in 0..num_chunks {
        let q_start = extra_tokens + ci * chunk_size;
        let q_len = chunk_size.min(s - q_start);
        let q_chunk = q.narrow(1, q_start, q_len)?;

        let k_start = q_start.saturating_sub(chunk_size / 2).max(extra_tokens);
        let k_len = (chunk_size * 2).min(s - k_start);
        let k_chunk = k.narrow(1, k_start, k_len)?;
        let v_chunk = v.narrow(1, k_start, k_len)?;

        let _out_chunk = candle_flash_attn::flash_attn(&q_chunk, &k_chunk, &v_chunk, scale, false)?;
    }
    dev.synchronize()?;
    let local_ms = t_local.elapsed().as_secs_f64() * 1000.0;
    println!("Local Window Branch Latency: {:.2} ms (Mean per chunk: {:.2} ms)",
        local_ms, local_ms / num_chunks as f64);

    let total_sparse_ms = pool_ms + anchor_ms + local_ms;
    let speedup = dense_ms / total_sparse_ms;

    println!("\n==================================================");
    println!("Phase 7B Performance Summary (H=4, S=121,041):");
    println!("  Dense FA-2 Latency       : {:.2} ms", dense_ms);
    println!("  Global Anchor Branch     : {:.2} ms", anchor_ms);
    println!("  Local Window Branch      : {:.2} ms", local_ms);
    println!("  Pooling Overhead         : {:.2} ms", pool_ms);
    println!("  Total Sparse Attention   : {:.2} ms", total_sparse_ms);
    println!("  Measured Speedup         : {:.2}x", speedup);
    println!("--------------------------------------------------");
    println!("Extrapolated 1080P 8s DiT Attention Time (47 layers x 14 groups):");
    println!("  Dense FA-2 Attention     : {:.1} s", dense_ms * 14.0 * 47.0 / 1000.0);
    println!("  Block-Sparse Attention   : {:.1} s", total_sparse_ms * 14.0 * 47.0 / 1000.0);
    println!("  Attention Time Saved     : {:.1} s / step",
        (dense_ms - total_sparse_ms) * 14.0 * 47.0 / 1000.0);
    println!("==================================================");

    Ok(())
}
