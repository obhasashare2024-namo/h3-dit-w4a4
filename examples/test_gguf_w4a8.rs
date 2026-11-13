use candle_core::{DType, Device, Tensor};
use h3_rs::weights::{SafeFile, QLinear, Placement};

fn main() -> anyhow::Result<()> {
    println!("=== Testing Direct GGUF W4A8 Loader and Inference on RTX 3070 ===");
    let dev = Device::new_cuda(0)?;
    let path = "/mnt/2tbnvS/models/minimax_h3_fl2va_w4a8.gguf";
    println!("Opening GGUF file: {path}");
    let sf = SafeFile::open(path)?;
    println!("GGUF file opened successfully!");
    println!("  Total tensors: {}", sf.tensors.len());
    println!("  Total metadata keys: {}", sf.metadata.len());
    if let Some(fmt) = sf.metadata.get("h3.quant.format") {
        println!("  h3.quant.format: {fmt}");
    }

    let prefix = "blocks.0.attn.qkv_proj";
    println!("Loading QLinear layer from GGUF: {prefix}");
    let qlinear = QLinear::load(&sf, prefix, Placement::Gpu, &dev, DType::BF16)?;
    println!("QLinear loaded from GGUF successfully!");
    println!("  in_features: {}", qlinear.in_features);
    println!("  out_features: {}", qlinear.out_features);

    println!("Testing forward pass with BF16 input on RTX 3070...");
    let x = Tensor::randn(0.0f32, 1.0, (1, qlinear.in_features), &dev)?.to_dtype(DType::BF16)?;
    
    // Warmup & test forward
    let y = qlinear.forward(&x)?;
    dev.synchronize()?;
    println!("Forward pass from GGUF weights succeeded!");
    println!("  Output shape: {:?}", y.dims());
    println!("  Output dtype: {:?}", y.dtype());

    // Check for NaNs
    let vals = y.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let nan_count = vals.iter().filter(|v| v.is_nan()).count();
    println!("  NaN count: {nan_count} / {}", vals.len());
    assert_eq!(nan_count, 0, "GGUF output contains NaNs!");

    // Benchmark 10 iterations
    println!("Benchmarking 10 forward iterations on GGUF weights...");
    let start = std::time::Instant::now();
    for _ in 0..10 {
        let _ = qlinear.forward(&x)?;
    }
    dev.synchronize()?;
    let elapsed = start.elapsed().as_secs_f64() / 10.0;
    println!("  Average latency per QLinear forward: {:.3} ms", elapsed * 1000.0);

    println!("=== Direct GGUF W4A8 Test PASSED with flying colors! ===");
    Ok(())
}
