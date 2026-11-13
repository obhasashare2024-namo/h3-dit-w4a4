use candle_core::{DType, Device, Tensor};
use h3_rs::weights::{SafeFile, QLinear, Placement};

fn main() -> anyhow::Result<()> {
    println!("=== Verifying Parity: Safetensors vs GGUF ===");
    let dev = Device::new_cuda(0)?;
    let p_st = "/mnt/2tbnvS/models/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors";
    let p_gguf = "/mnt/2tbnvS/models/minimax_h3_fl2va_w4a8.gguf";

    let sf_st = SafeFile::open(p_st)?;
    let sf_gguf = SafeFile::open(p_gguf)?;

    let prefix = "blocks.0.attn.qkv_proj";
    let q_st = QLinear::load(&sf_st, prefix, Placement::Gpu, &dev, DType::BF16)?;
    let q_gguf = QLinear::load(&sf_gguf, prefix, Placement::Gpu, &dev, DType::BF16)?;

    // Identical input
    let x = Tensor::randn(0.0f32, 1.0, (1, q_st.in_features), &dev)?.to_dtype(DType::BF16)?;

    let y_st = q_st.forward(&x)?;
    let y_gguf = q_gguf.forward(&x)?;
    dev.synchronize()?;

    let diff = (&y_st.to_dtype(DType::F32)? - &y_gguf.to_dtype(DType::F32)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("Maximum absolute difference |y_st - y_gguf|: {diff:.6e}");
    assert_eq!(diff, 0.0, "Parity check failed: output difference > 0!");
    println!("=== 100% BIT-EXACT PARITY VERIFIED! ===");
    Ok(())
}
