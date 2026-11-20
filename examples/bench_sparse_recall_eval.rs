use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    println!("=================================================================");
    println!("=== DEF-081: Softmax Query Recall Evaluation (rc001 Gate) ===");
    println!("=================================================================");

    let s = 121_041usize;
    let d = 128usize;
    let extra_tokens = 4_761usize;
    let video_tokens = 116_280usize;
    let (t_len, h_len, w_len) = (57usize, 34usize, 60usize);
    let frame_patches = h_len * w_len; // 2040

    println!("Grid Layout: Extra={}, Video={} (T={}, H={}, W={})",
        extra_tokens, video_tokens, t_len, h_len, w_len);

    let q = Tensor::randn(0f32, 1.0, (s, 1, d), &dev)?.to_dtype(DType::F16)?;
    let k_offset = Tensor::randn(0f32, 1.5, (1, 1, d), &dev)?.to_dtype(DType::F16)?.broadcast_as((s, 1, d))?;
    let k = (Tensor::randn(0f32, 1.0, (s, 1, d), &dev)?.to_dtype(DType::F16)? + k_offset)?;

    let mut rng = 123456789u64;
    let num_queries = 256usize;
    let mut sample_queries = Vec::with_capacity(num_queries);
    for _ in 0..num_queries {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let q_idx = extra_tokens + ((rng >> 32) as usize % video_tokens);
        sample_queries.push(q_idx);
    }

    let q_sampled_vec: Vec<Tensor> = sample_queries.iter()
        .map(|&idx| q.narrow(0, idx, 1).unwrap())
        .collect();
    let q_sampled = Tensor::cat(&q_sampled_vec, 0)?; // [num_queries, 1, d]

    let scale = 1.0 / (d as f32).sqrt();

    let q_flat = q_sampled.reshape((num_queries, d))?.to_dtype(DType::F32)?;
    let k_flat = k.reshape((s, d))?.to_dtype(DType::F32)?;

    println!("Computing Full Softmax LSE across S={} for {} queries...", s, num_queries);
    let t0 = Instant::now();
    let scores = (q_flat.matmul(&k_flat.t()?)? * scale as f64)?; // [num_queries, s]
    let max_score = scores.max_keepdim(1)?;
    let shifted = scores.broadcast_sub(&max_score)?;
    let exp_scores = shifted.exp()?;
    let sum_exp_full = exp_scores.sum_keepdim(1)?; // [num_queries, 1]
    dev.synchronize()?;
    println!("Full LSE computed in {:.2} ms", t0.elapsed().as_secs_f64() * 1000.0);

    let window_configs = [
        ("Tight (dt=1, dh=2, dw=3)", 1, 2, 3),
        ("Medium (dt=2, dh=3, dw=4)", 2, 3, 4),
        ("Wide (dt=3, dh=4, dw=6)", 3, 4, 6),
    ];

    let exp_scores_vec = exp_scores.to_vec2::<f32>()?;
    let sum_exp_full_vec = sum_exp_full.flatten_all()?.to_vec1::<f32>()?;

    for &(cfg_name, dt, dh, dw) in &window_configs {
        let mut recalls = Vec::with_capacity(num_queries);
        let mut active_tokens_count = 0usize;

        for (qi, &q_idx) in sample_queries.iter().enumerate() {
            let v_idx = q_idx - extra_tokens;
            let qt = v_idx / frame_patches;
            let rem = v_idx % frame_patches;
            let qh = rem / w_len;
            let qw = rem % w_len;

            let mut sum_exp_mask = 0.0f64;
            let mut num_active = 0usize;

            // 1. Extra tokens (Text, Cond, Audio) are 100% retained
            for j in 0..extra_tokens {
                sum_exp_mask += exp_scores_vec[qi][j] as f64;
                num_active += 1;
            }

            // 2. 3D Window + Temporal Tube over video tokens
            for jt in 0..t_len {
                let is_in_t_window = (jt as isize - qt as isize).abs() <= dt;
                let h_min = if is_in_t_window { (qh as isize - dh).max(0) as usize } else { (qh as isize - 1).max(0) as usize };
                let h_max = if is_in_t_window { (qh as isize + dh).min(h_len as isize - 1) as usize } else { (qh as isize + 1).min(h_len as isize - 1) as usize };
                let w_min = if is_in_t_window { (qw as isize - dw).max(0) as usize } else { (qw as isize - 1).max(0) as usize };
                let w_max = if is_in_t_window { (qw as isize + dw).min(w_len as isize - 1) as usize } else { (qw as isize + 1).min(w_len as isize - 1) as usize };

                for jh in h_min..=h_max {
                    for jw in w_min..=w_max {
                        let j_token = extra_tokens + jt * frame_patches + jh * w_len + jw;
                        sum_exp_mask += exp_scores_vec[qi][j_token] as f64;
                        num_active += 1;
                    }
                }
            }

            let recall = (sum_exp_mask / sum_exp_full_vec[qi] as f64) as f32;
            recalls.push(recall);
            active_tokens_count += num_active;
        }

        recalls.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_recall = recalls[num_queries / 2];
        let p5_recall = recalls[(num_queries as f32 * 0.05) as usize];
        let p1_recall = recalls[0];
        let avg_active = active_tokens_count / num_queries;
        let sparsity = 1.0 - (avg_active as f64 / s as f64);

        println!("\n--- Configuration: {} ---", cfg_name);
        println!("  Active Tokens / Query: {} / {} (Density: {:.2}%, Sparsity: {:.2}%)",
            avg_active, s, (1.0 - sparsity) * 100.0, sparsity * 100.0);
        println!("  Softmax Recall Median : {:.4} (Gate >= 0.9700)", median_recall);
        println!("  Softmax Recall p5     : {:.4} (Gate >= 0.9000)", p5_recall);
        println!("  Softmax Recall min (p1): {:.4}", p1_recall);

        let pass_gate = median_recall >= 0.97 && p5_recall >= 0.90;
        println!("  ===> GATE STATUS: [{}]", if pass_gate { "PASS" } else { "NEED WIDER WINDOW" });
    }

    Ok(())
}
