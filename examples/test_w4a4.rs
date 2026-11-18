use candle_core::{DType, Device};
use h3_rs::weights::{Placement, QLinear, SafeFile};

fn main() -> anyhow::Result<()> {
    println!("=== Testing QLinear Parity vs PyTorch Reference ===");
    let dev = Device::new_cuda(0)?;
    let model_path = "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.safetensors";
    let sf = SafeFile::open(model_path)?;

    let prefix = "blocks.0.attn.qkv_proj";
    let qlinear = QLinear::load(&sf, prefix, Placement::Gpu, &dev, DType::BF16)?;

    let parity_path = "/mnt/2tbnvS/tmp/test_parity_tensors.safetensors";
    let pf = SafeFile::open(parity_path)?;
    let x = pf.load_as("X", &dev, DType::BF16)?;
    let y_ref = pf.load_as("Y_ref", &dev, DType::F32)?;

    println!("Running QLinear forward on GPU...");
    let y_rust = qlinear.forward(&x)?.to_dtype(DType::F32)?;
    dev.synchronize()?;

    let v_rust = y_rust.flatten_all()?.to_vec1::<f32>()?;
    let v_ref = y_ref.flatten_all()?.to_vec1::<f32>()?;

    let mut dot = 0.0f64;
    let mut norm_rust = 0.0f64;
    let mut norm_ref = 0.0f64;
    let mut max_err = 0.0f32;

    for (r, f) in v_rust.iter().zip(v_ref.iter()) {
        dot += (*r as f64) * (*f as f64);
        norm_rust += (*r as f64) * (*r as f64);
        norm_ref += (*f as f64) * (*f as f64);
        let diff = (r - f).abs();
        if diff > max_err {
            max_err = diff;
        }
    }

    let cos_sim = dot / (norm_rust.sqrt() * norm_ref.sqrt());
    println!("=== NUMERICAL PARITY RESULTS ===");
    println!("Cosine Similarity vs PyTorch: {:.6}", cos_sim);
    println!("Max Absolute Error: {:.4}", max_err);
    println!("Rust output std: {:.4}, PyTorch ref std: {:.4}", (norm_rust / v_rust.len() as f64).sqrt(), (norm_ref / v_ref.len() as f64).sqrt());

    for i in 0..10 {
        println!("  [{}] rust: {:.3}, ref: {:.3}", i, v_rust[i], v_ref[i]);
    }

    Ok(())
}
