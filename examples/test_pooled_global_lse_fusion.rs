use std::time::Instant;
use candle_core::{DType, Device, Result, Tensor};

fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32> {
    let dot = (a * b)?.sum_all()?.to_scalar::<f32>()?;
    let norm_a = (a * a)?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let norm_b = (b * b)?.sum_all()?.to_scalar::<f32>()?.sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }
    Ok(dot / (norm_a * norm_b))
}

fn rel_l2_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let diff = (a - b)?;
    let diff_norm = (&diff * &diff)?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let ref_norm = (a * a)?.sum_all()?.to_scalar::<f32>()?.sqrt();
    if ref_norm == 0.0 {
        return Ok(0.0);
    }
    Ok(diff_norm / ref_norm)
}

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    println!("=================================================================");
    println!("=== DEF-082: 64-Tile Pooled Global Branch & LSE Fusion Oracle ===");
    println!("=================================================================");

    let s = 121_041usize;
    let d = 128usize;
    let extra_tokens = 4_761usize;
    let video_tokens = 116_280usize;
    let tile_size = 64usize;
    let full_tiles = video_tokens / tile_size; // 1816 tiles
    let rem_tokens = video_tokens % tile_size; // 56 tokens
    let num_tiles = if rem_tokens > 0 { full_tiles + 1 } else { full_tiles }; // 1817 tiles

    println!("Config: S={}, Extra={}, Video={}, TileSize={}, FullTiles={}, RemTokens={}, NumTiles={}",
        s, extra_tokens, video_tokens, tile_size, full_tiles, rem_tokens, num_tiles);

    // 1. Create Tile-Major Structured Video Tokens
    println!("Creating tile-major structured Q, K, V...");
    let tile_k = Tensor::randn(0f32, 1.0, (num_tiles, d), &dev)?.to_dtype(DType::F32)?;
    let tile_v = Tensor::randn(0f32, 1.0, (num_tiles, d), &dev)?.to_dtype(DType::F32)?;

    // Expand tiles into video tokens: repeat each tile vector tile_size times
    let main_video_k = tile_k.narrow(0, 0, full_tiles)?
        .unsqueeze(1)?
        .broadcast_as((full_tiles, tile_size, d))?
        .contiguous()?
        .reshape((full_tiles * tile_size, d))?;
    let main_video_v = tile_v.narrow(0, 0, full_tiles)?
        .unsqueeze(1)?
        .broadcast_as((full_tiles, tile_size, d))?
        .contiguous()?
        .reshape((full_tiles * tile_size, d))?;

    let (video_k, video_v) = if rem_tokens > 0 {
        let rem_k = tile_k.narrow(0, full_tiles, 1)?
            .broadcast_as((rem_tokens, d))?
            .contiguous()?;
        let rem_v = tile_v.narrow(0, full_tiles, 1)?
            .broadcast_as((rem_tokens, d))?
            .contiguous()?;
        (Tensor::cat(&[&main_video_k, &rem_k], 0)?, Tensor::cat(&[&main_video_v, &rem_v], 0)?)
    } else {
        (main_video_k, main_video_v)
    };

    // Add fine-grained intra-tile perturbation (std = 0.1)
    let noise_k = Tensor::randn(0f32, 0.1, video_k.shape(), &dev)?.to_dtype(DType::F32)?;
    let noise_v = Tensor::randn(0f32, 0.1, video_v.shape(), &dev)?.to_dtype(DType::F32)?;
    let video_k = (video_k + noise_k)?;
    let video_v = (video_v + noise_v)?;

    // Extra tokens (Text, Cond, Audio)
    let extra_k = Tensor::randn(0f32, 1.0, (extra_tokens, d), &dev)?.to_dtype(DType::F32)?;
    let extra_v = Tensor::randn(0f32, 1.0, (extra_tokens, d), &dev)?.to_dtype(DType::F32)?;

    let k_all = Tensor::cat(&[&extra_k, &video_k], 0)?; // [S, D]
    let v_all = Tensor::cat(&[&extra_v, &video_v], 0)?; // [S, D]

    // Sample 64 test queries with realistic diffusion semantics (aligned to local patch + text prompt)
    let num_queries = 64usize;
    let mut q_vecs = Vec::with_capacity(num_queries);
    let mut sample_centers = Vec::with_capacity(num_queries);

    let text_anchor = extra_k.narrow(0, 0, 1)?; // primary prompt token
    for qi in 0..num_queries {
        let center = extra_tokens + qi * (video_tokens / num_queries);
        sample_centers.push(center);
        let patch_k = k_all.narrow(0, center, 1)?;
        // Query has semantic affinity with local patch + text prompt + minor noise
        let noise_q = Tensor::randn(0f32, 0.2, (1, d), &dev)?.to_dtype(DType::F32)?;
        let q_i = ((&patch_k * 1.5)? + (&text_anchor * 1.0)? + &noise_q)?;
        q_vecs.push(q_i);
    }
    let q_sample = Tensor::cat(&q_vecs, 0)?; // [num_queries, d]

    let scale = 1.0 / (d as f32).sqrt();

    // 2. Ground Truth Full Dense Attention
    println!("Computing Ground Truth Full Dense Attention [{} x {}]...", num_queries, s);
    let t0 = Instant::now();
    let full_scores = (q_sample.matmul(&k_all.t()?)? * scale as f64)?; // [num_queries, S]
    let full_max = full_scores.max_keepdim(1)?;
    let full_exp = full_scores.broadcast_sub(&full_max)?.exp()?;
    let full_sum = full_exp.sum_keepdim(1)?;
    let full_lse = full_max.broadcast_add(&full_sum.log()?)?; // [num_queries, 1]
    let out_dense = full_exp.matmul(&v_all)?.broadcast_div(&full_sum)?;
    dev.synchronize()?;
    let dense_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("Ground Truth computed in {:.2} ms", dense_ms);

    // 3. Pooled Global Branch: compute mean K and mean V per tile
    println!("Computing 64-Tile Pooled Global representations...");
    let t1 = Instant::now();

    let full_k_reshaped = video_k.narrow(0, 0, full_tiles * tile_size)?.reshape((full_tiles, tile_size, d))?;
    let full_v_reshaped = video_v.narrow(0, 0, full_tiles * tile_size)?.reshape((full_tiles, tile_size, d))?;
    let pooled_k_main = full_k_reshaped.mean(1)?; // [full_tiles, d]
    let pooled_v_main = full_v_reshaped.mean(1)?; // [full_tiles, d]

    let (pooled_k, pooled_v) = if rem_tokens > 0 {
        let rem_k = video_k.narrow(0, full_tiles * tile_size, rem_tokens)?.mean(0)?.reshape((1, d))?;
        let rem_v = video_v.narrow(0, full_tiles * tile_size, rem_tokens)?.mean(0)?.reshape((1, d))?;
        (Tensor::cat(&[&pooled_k_main, &rem_k], 0)?, Tensor::cat(&[&pooled_v_main, &rem_v], 0)?)
    } else {
        (pooled_k_main, pooled_v_main)
    };

    // Combine extra tokens + pooled tiles for global anchor bank
    let anchors_k = Tensor::cat(&[&extra_k, &pooled_k], 0)?; // [extra + num_tiles, d]
    let anchors_v = Tensor::cat(&[&extra_v, &pooled_v], 0)?; // [extra + num_tiles, d]

    // Base bias for pooled tiles: +ln(tile_len) for tiles, +0 for extra tokens
    let mut base_bias_vec = vec![0.0f32; extra_tokens];
    let ln_full = (tile_size as f32).ln();
    base_bias_vec.extend(vec![ln_full; full_tiles]);
    if rem_tokens > 0 {
        base_bias_vec.push((rem_tokens as f32).ln());
    }

    // 4. Two-Branch LSE Fusion with Double-Counting Mask
    let local_window = 4096usize;
    let mut fused_outs = Vec::with_capacity(num_queries);
    let mut recalls = Vec::with_capacity(num_queries);
    let full_lse_vec = full_lse.flatten_all()?.to_vec1::<f32>()?;

    for qi in 0..num_queries {
        let q_vec = q_sample.narrow(0, qi, 1)?; // [1, d]
        let center = sample_centers[qi];
        let start = center.saturating_sub(local_window / 2).max(extra_tokens);
        let end = (start + local_window).min(s);
        let local_len = end - start;

        // Local branch
        let k_local = k_all.narrow(0, start, local_len)?;
        let v_local = v_all.narrow(0, start, local_len)?;

        let local_scores = (q_vec.matmul(&k_local.t()?)? * scale as f64)?; // [1, local_len]
        let local_max = local_scores.max_keepdim(1)?;
        let local_exp = local_scores.broadcast_sub(&local_max)?.exp()?;
        let local_sum = local_exp.sum_keepdim(1)?;
        let local_lse = local_max.broadcast_add(&local_sum.log()?)?; // [1, 1]
        let out_local_unnorm = local_exp.matmul(&v_local)?; // [1, d]

        // Global branch: mask out tiles already covered by local window
        let tile_start = (start - extra_tokens) / tile_size;
        let tile_end = ((end - extra_tokens) + tile_size - 1) / tile_size;

        let mut query_bias_vec = base_bias_vec.clone();
        for t_idx in tile_start..tile_end.min(num_tiles) {
            query_bias_vec[extra_tokens + t_idx] = -1e9f32; // mask covered tile
        }
        let bias_t = Tensor::from_vec(query_bias_vec, (1, extra_tokens + num_tiles), &dev)?;

        let global_scores = (q_vec.matmul(&anchors_k.t()?)? * scale as f64)?.broadcast_add(&bias_t)?;
        let global_max = global_scores.max_keepdim(1)?;
        let global_exp = global_scores.broadcast_sub(&global_max)?.exp()?;
        let global_sum = global_exp.sum_keepdim(1)?;
        let global_lse = global_max.broadcast_add(&global_sum.log()?)?; // [1, 1]
        let out_global_unnorm = global_exp.matmul(&anchors_v)?; // [1, d]

        // LSE Fusion
        let m_combined = local_lse.maximum(&global_lse)?;
        let l_scale = local_lse.broadcast_sub(&m_combined)?.exp()?;
        let g_scale = global_lse.broadcast_sub(&m_combined)?.exp()?;
        let denom = l_scale.broadcast_add(&g_scale)?;

        let l_weight = local_max.broadcast_sub(&m_combined)?.exp()?;
        let g_weight = global_max.broadcast_sub(&m_combined)?.exp()?;

        let term_l = out_local_unnorm.broadcast_mul(&l_weight)?;
        let term_g = out_global_unnorm.broadcast_mul(&g_weight)?;
        let num = (&term_l + &term_g)?;
        let denom_b = denom.broadcast_as(num.shape())?;
        let out_fused_i = (&num / &denom_b)?;
        fused_outs.push(out_fused_i);

        // Softmax Query Recall: exp(fused_lse - full_lse)
        let fused_lse = m_combined.broadcast_add(&denom.log()?)?.flatten_all()?.to_vec1::<f32>()?[0];
        let recall = (fused_lse - full_lse_vec[qi]).exp();
        recalls.push(recall);
    }
    dev.synchronize()?;
    let fused_ms = t1.elapsed().as_secs_f64() * 1000.0;
    println!("Pooled Global + Local Window Fusion computed in {:.2} ms", fused_ms);

    let out_fused = Tensor::cat(&fused_outs, 0)?;

    // 5. Precision Check
    let cos = cosine_similarity(&out_dense, &out_fused)?;
    let rel_diff = rel_l2_diff(&out_dense, &out_fused)?;

    recalls.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_recall = recalls[num_queries / 2];
    let p5_recall = recalls[(num_queries as f32 * 0.05) as usize];
    let min_recall = recalls[0];

    let total_active_tokens = local_window + extra_tokens + (num_tiles - (local_window / tile_size));
    let sparsity = 1.0 - (total_active_tokens as f64 / s as f64);

    println!("--------------------------------------------------");
    println!("Parity Metrics:");
    println!("  Active Tokens / Query: {} / {} (Density: {:.2}%, Sparsity: {:.2}%)",
        total_active_tokens, s, (1.0 - sparsity) * 100.0, sparsity * 100.0);
    println!("  Cosine Similarity    : {:.6} (Gate >= 0.9700)", cos);
    println!("  Relative L2 Error    : {:.4}% (Gate < 10.0%)", rel_diff * 100.0);
    println!("  Softmax Recall Median: {:.6} (Gate >= 0.9700)", median_recall);
    println!("  Softmax Recall p5    : {:.6} (Gate >= 0.9000)", p5_recall);
    println!("  Softmax Recall min   : {:.6}", min_recall);
    println!("--------------------------------------------------");

    assert!(cos >= 0.97, "Cosine similarity failed gate! cos={:.6}", cos);
    assert!(rel_diff < 0.10, "Relative L2 diff failed gate! rel_diff={:.4}%", rel_diff * 100.0);
    assert!(median_recall >= 0.97, "Median recall failed gate! median={:.6}", median_recall);
    assert!(p5_recall >= 0.90, "P5 recall failed gate! p5={:.6}", p5_recall);
    println!("===> STATUS: [PASS] DEF-082 VERIFIED");

    Ok(())
}
