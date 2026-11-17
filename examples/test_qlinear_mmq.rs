use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use h3_rs::weights::{GgmlType, Held, QFormat, QLinear, Raw};

fn main() -> Result<()> {
    println!("========================================================");
    println!("  DEF-028: QLinear MMQ Integration & A/B Fallback Test  ");
    println!("========================================================");

    let dev = Device::new_cuda(0)?;
    let in_features = 5376usize;
    let out_features = 256usize;
    let m = 16usize;

    println!("Shape: Tokens M={}, In K={}, Out N={}", m, in_features, out_features);

    // 1. Generate valid Q3_K raw block buffer
    let num_blocks = (out_features * in_features) / 256;
    let raw_bytes_len = num_blocks * 110;
    let mut raw = vec![0x55u8; raw_bytes_len];
    for b in 0..num_blocks {
        let base = b * 110;
        raw[base + 108] = 0x66;
        raw[base + 109] = 0x2A; // 0.05 in f16
        for s in 0..12 {
            raw[base + 96 + s] = (s as u8 * 17 + b as u8 * 7) ^ 0xAA;
        }
    }
    let raw_t = Tensor::from_vec(raw.clone(), (raw_bytes_len,), &dev)?;
    let raw_weight = Raw { w: raw_t, bs: None };

    // 2. Build synthetic QLinear with QFormat::GgufQuant { gtype: GgmlType::Q3K }
    let qlinear = QLinear {
        name: "test_q3k_linear".to_string(),
        fmt: QFormat::GgufQuant { gtype: GgmlType::Q3K },
        weight: Held::Dev(raw_weight.w.clone()),
        scale: None,
        block_scale: None,
        pre_quant_scale: None,
        bias: None,
        loras: vec![],
        out_features,
        in_features,
        interleaved: false,
    };

    let x_bf16 = Tensor::randn(0.0f32, 1.0, (m, in_features), &dev)?.to_dtype(DType::BF16)?;

    // 3. Test Path A: Fallback to DequantArena (H3_MMQ=0)
    std::env::set_var("H3_MMQ", "0");
    let p_deq = qlinear.prepare(raw_weight.clone(), &dev, DType::BF16)?;
    let is_deq = matches!(p_deq, h3_rs::weights::Prepared::Deq(_));
    println!("Path A (H3_MMQ=0): Prepared variant is Deq: {}", is_deq);
    assert!(is_deq, "Expected Prepared::Deq when H3_MMQ=0");

    let y_deq = qlinear.run(&x_bf16, &p_deq, 1024)?;
    dev.synchronize()?;

    // 4. Test Path B: Native MMQ Direct (H3_MMQ=1)
    std::env::set_var("H3_MMQ", "1");
    // Force OnceLock reset in new process or test directly
    let p_mmq = h3_rs::weights::Prepared::Mmq(raw_weight.clone());
    println!("Path B (MMQ): Dispatching via apply_mmq...");
    let y_mmq = qlinear.run(&x_bf16, &p_mmq, 1024)?;
    dev.synchronize()?;

    // 5. Compare Bitwise & Cosine
    let y_deq_f32 = y_deq.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let y_mmq_f32 = y_mmq.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

    let mut max_abs_diff = 0.0f32;
    let mut dot_product = 0.0f64;
    let mut norm_deq = 0.0f64;
    let mut norm_mmq = 0.0f64;
    let mut has_nan = false;
    let mut has_inf = false;

    for i in 0..y_deq_f32.len() {
        let r = y_deq_f32[i];
        let q = y_mmq_f32[i];

        if r.is_nan() || q.is_nan() { has_nan = true; }
        if r.is_infinite() || q.is_infinite() { has_inf = true; }

        let diff = (r - q).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }

        dot_product += (r as f64) * (q as f64);
        norm_deq += (r as f64) * (r as f64);
        norm_mmq += (q as f64) * (q as f64);
    }

    let cosine = dot_product / (norm_deq.sqrt() * norm_mmq.sqrt());
    let rel_diff = max_abs_diff / (y_deq_f32.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-5));

    println!("\n=== QLINEAR END-TO-END QUALITY GATES ===");
    println!("  Total output elements: {}", y_deq_f32.len());
    println!("  Max Absolute Diff:     {:.6}", max_abs_diff);
    println!("  Relative Diff:         {:.6} (Gate: < 0.05)", rel_diff);
    println!("  Cosine Similarity:     {:.6} (Gate: > 0.999)", cosine);
    println!("  Has NaN:               {} (Gate: false)", has_nan);
    println!("  Has Inf:               {} (Gate: false)", has_inf);

    assert!(!has_nan, "NaN in output!");
    assert!(!has_inf, "Inf in output!");
    assert!(cosine > 0.999, "Cosine similarity failed gate!");
    assert!(rel_diff < 0.05, "Relative difference failed gate!");

    println!("\n🎉 DEF-028: QLinear MMQ Integration & Fallback VERIFIED 100%!");
    Ok(())
}
