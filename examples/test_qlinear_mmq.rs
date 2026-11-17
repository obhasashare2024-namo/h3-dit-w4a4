use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use h3_rs::weights::{GgmlType, Held, QFormat, QLinear, Raw};

fn test_case(name: &str, gtype: GgmlType, bytes_per_block: usize, block_size: usize) -> Result<()> {
    println!("\n--------------------------------------------------------");
    println!("  Testing QLinear MMQ Integration: {:?} ({})", gtype, name);
    println!("--------------------------------------------------------");

    let dev = Device::new_cuda(0)?;
    let in_features = 5376usize;
    let out_features = 256usize;
    let m = 16usize;

    let num_blocks = (out_features * in_features) / block_size;
    let raw_bytes_len = num_blocks * bytes_per_block;
    let mut raw = vec![0x55u8; raw_bytes_len];
    for b in 0..num_blocks {
        let base = b * bytes_per_block;
        if bytes_per_block == 110 {
            raw[base + 108] = 0x66; raw[base + 109] = 0x2A;
            for s in 0..12 { raw[base + 96 + s] = (s as u8 * 17 + b as u8 * 7) ^ 0xAA; }
        } else if bytes_per_block == 144 {
            raw[base + 0] = 0x66; raw[base + 1] = 0x2A;
            raw[base + 2] = 0x11; raw[base + 3] = 0x22;
            for s in 0..12 { raw[base + 4 + s] = (s as u8 * 13 + b as u8 * 5) ^ 0x55; }
            for q in 0..128 { raw[base + 16 + q] = (q as u8 * 19) ^ 0x33; }
        } else if bytes_per_block == 176 {
            raw[base + 0] = 0x66; raw[base + 1] = 0x2A;
            raw[base + 2] = 0x11; raw[base + 3] = 0x22;
            for s in 0..12 { raw[base + 4 + s] = (s as u8 * 13 + b as u8 * 5) ^ 0x55; }
            for q in 0..32 { raw[base + 16 + q] = (q as u8 * 19) ^ 0x33; }
        } else if bytes_per_block == 210 {
            for q in 0..128 { raw[base + q] = ((q as u8 * 17) ^ 0x93) & 0xFF; }
            for q in 0..64 { raw[base + 128 + q] = ((q as u8 * 31) ^ 0x47) & 0xFF; }
            for s in 0..16 { raw[base + 192 + s] = (((s as i8 * 7) ^ 0x23) as u8) & 0xFF; }
            raw[base + 208] = 0x66; raw[base + 209] = 0x2A;
        } else if bytes_per_block == 34 {
            raw[base + 0] = 0x66; raw[base + 1] = 0x2A;
            for q in 0..32 { raw[base + 2 + q] = ((q as u8 * 17 + b as u8 * 3) ^ 0x33) as u8; }
        }
    }
    let raw_t = Tensor::from_vec(raw, (raw_bytes_len,), &dev)?;
    let raw_weight = Raw { w: raw_t, bs: None };

    let qlinear = QLinear {
        name: format!("test_{}_linear", name),
        fmt: QFormat::GgufQuant { gtype },
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

    // Ground truth: dequant_raw + cuBLAS
    let p_deq = h3_rs::weights::Prepared::Deq(qlinear.dequant_raw(&raw_weight, &dev, DType::BF16)?);
    let y_deq = qlinear.run(&x_bf16, &p_deq, 1024)?;
    dev.synchronize()?;

    // MMQ Direct
    let p_mmq = h3_rs::weights::Prepared::Mmq(raw_weight.clone());
    let y_mmq = qlinear.run(&x_bf16, &p_mmq, 1024)?;
    dev.synchronize()?;

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
        if diff > max_abs_diff { max_abs_diff = diff; }

        dot_product += (r as f64) * (q as f64);
        norm_deq += (r as f64) * (r as f64);
        norm_mmq += (q as f64) * (q as f64);
    }

    let cosine = dot_product / (norm_deq.sqrt() * norm_mmq.sqrt());
    let rel_diff = max_abs_diff / (y_deq_f32.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-5));

    println!("  Total output elements: {}", y_deq_f32.len());
    println!("  Max Absolute Diff:     {:.6}", max_abs_diff);
    println!("  Relative Diff:         {:.6} (Gate: < 0.05)", rel_diff);
    println!("  Cosine Similarity:     {:.6} (Gate: > 0.999)", cosine);
    println!("  Has NaN / Inf:         {} / {}", has_nan, has_inf);

    assert!(!has_nan, "NaN in output!");
    assert!(!has_inf, "Inf in output!");
    assert!(cosine > 0.999, "Cosine similarity failed gate!");
    assert!(rel_diff < 0.05, "Relative difference failed gate!");

    println!("  [PASS] {} MMQ Verification Successful!", name);
    Ok(())
}

fn main() -> Result<()> {
    println!("========================================================");
    println!("  DEF-038: QLinear MMQ Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 Penta Test ");
    println!("========================================================");

    test_case("Q3_K", GgmlType::Q3K, 110, 256)?;
    test_case("Q4_K", GgmlType::Q4K, 144, 256)?;
    test_case("Q5_K", GgmlType::Q5K, 176, 256)?;
    test_case("Q6_K", GgmlType::Q6K, 210, 256)?;
    test_case("Q8_0", GgmlType::Q8_0, 34, 32)?;

    println!("\n🎉 ALL QLinear MMQ PENTA INTEGRATION TESTS PASSED 100%!");
    Ok(())
}
