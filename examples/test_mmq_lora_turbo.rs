use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use h3_rs::lora::Lora;
use h3_rs::weights::{Placement, QLinear, SafeFile, LoraDelta};

fn main() -> Result<()> {
    println!("=== Testing GGUF MMQ + Turbo LoRA Integration on RTX 3070 ===");
    let dev = Device::new_cuda(0)?;

    let lora_path = "/mnt/2tbnvS/models/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors";
    let gguf_path = "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b-Q3_K_M.gguf";

    println!("[1] Opening Turbo LoRA: {lora_path}");
    let lora = Lora::open(lora_path)?;
    println!("    LoRA device bytes: {:.2} MB", lora.device_bytes() as f64 / 1e6);

    println!("[2] Opening GGUF base model: {gguf_path}");
    let sf = SafeFile::open(gguf_path)?;

    let prefix = "blocks.0.attn.out_proj";
    println!("[3] Loading base QLinear: {prefix}");
    let mut qlinear = QLinear::load(&sf, prefix, Placement::Gpu, &dev, DType::BF16)?;
    println!("    Base QLinear format loaded");
    println!("    in_features: {}, out_features: {}", qlinear.in_features, qlinear.out_features);

    // Attach LoRA delta
    println!("[4] Fetching LoRA delta for {prefix}...");
    if let Some(d) = lora.delta(prefix, &dev)? {
        println!("    LoRA A: {:?}, B: {:?}, scale: {}", d.a.shape(), d.b.shape(), d.scale);
        qlinear.loras.push(LoraDelta {
            a: d.a,
            b: d.b,
            scale: d.scale,
        });
        println!("    LoRA attached to QLinear successfully!");
    } else {
        panic!("LoRA delta not found for {prefix}!");
    }

    println!("[5] Running QLinear forward pass with MMQ + LoRA on RTX 3070...");
    let x = Tensor::randn(0.0f32, 1.0, (1, 64, qlinear.in_features), &dev)?.to_dtype(DType::BF16)?;
    let y = qlinear.forward(&x)?;
    dev.synchronize()?;

    println!("    Output shape: {:?}", y.dims());
    println!("    Output dtype: {:?}", y.dtype());

    // Quality check
    let vals = y.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let nan_count = vals.iter().filter(|v| v.is_nan()).count();
    let inf_count = vals.iter().filter(|v| v.is_infinite()).count();
    println!("    NaN count: {nan_count} / {}", vals.len());
    println!("    Inf count: {inf_count} / {}", vals.len());
    assert_eq!(nan_count, 0, "Output contains NaNs!");
    assert_eq!(inf_count, 0, "Output contains Infs!");

    println!("🎉 DEF-047 GGUF MMQ + TURBO LORA INTEGRATION PASSED 100%!");
    Ok(())
}
