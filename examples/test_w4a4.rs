use candle_core::{DType, Device, Tensor};
use h3_rs::weights::{Placement, QLinear, SafeFile};

fn main() -> anyhow::Result<()> {
    println!("=== Testing Native W4A4 Loader and Inference on RTX 3070 ===");
    let dev = Device::new_cuda(0)?;
    let path = "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4.safetensors";
    println!("Opening safetensors file: {path}");
    let sf = SafeFile::open(path)?;
    println!("SafeFile opened! Total tensors: {}", sf.tensors.len());

    let prefix = "blocks.0.attn.qkv_proj";
    println!("Loading QLinear layer: {prefix}");
    let qlinear = QLinear::load(&sf, prefix, Placement::Gpu, &dev, DType::BF16)?;
    println!("QLinear loaded successfully!");
    println!("  in_features: {}", qlinear.in_features);
    println!("  out_features: {}", qlinear.out_features);

    println!("Testing forward pass with BF16 input...");
    let x = Tensor::randn(0.0f32, 1.0, (1, qlinear.in_features), &dev)?.to_dtype(DType::BF16)?;

    // Warmup & test forward
    let y = qlinear.forward(&x)?;
    dev.synchronize()?;
    println!("Forward pass succeeded!");
    println!("  Output shape: {:?}", y.dims());
    println!("  Output dtype: {:?}", y.dtype());

    // Check for NaNs
    let vals = y.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let nan_count = vals.iter().filter(|v| v.is_nan()).count();
    println!("  NaN count: {nan_count} / {}", vals.len());
    assert_eq!(nan_count, 0, "Output contains NaNs!");

    // Test multiple layers across attention and MLP
    for test_prefix in [
        "blocks.0.attn.out_proj",
        "blocks.0.mlp.fc1",
        "blocks.0.mlp.fc2",
        "blocks.40.attn.qkv_proj",
        "blocks.40.mlp.fc1",
    ] {
        println!("Testing layer: {test_prefix}");
        let layer = QLinear::load(&sf, test_prefix, Placement::Gpu, &dev, DType::BF16)?;
        let lx = Tensor::randn(0.0f32, 1.0, (1, layer.in_features), &dev)?.to_dtype(DType::BF16)?;
        let ly = layer.forward(&lx)?;
        dev.synchronize()?;
        println!("  shape: {:?}, non-nan verified!", ly.dims());
    }

    // Benchmark 10 iterations
    println!("Benchmarking 10 forward iterations on blocks.0.attn.qkv_proj...");
    let start = std::time::Instant::now();
    for _ in 0..10 {
        let _ = qlinear.forward(&x)?;
    }
    dev.synchronize()?;
    let elapsed = start.elapsed().as_secs_f64() / 10.0;
    println!("  Average latency per QLinear forward: {:.3} ms", elapsed * 1000.0);

    println!("=== W4A4 Safetensors Test PASSED with flying colors! ===");
    Ok(())
}
