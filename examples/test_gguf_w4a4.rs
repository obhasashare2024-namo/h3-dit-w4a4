use candle_core::{DType, Device, Tensor};
use h3_rs::weights::{Placement, QLinear, SafeFile};

fn main() -> anyhow::Result<()> {
    println!("=== Testing Native Ampere INT4 Tensor Core W4A4 in GGUF on RTX 3070 ===");
    let dev = Device::new_cuda(0)?;
    let path = "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.gguf";
    println!("Opening GGUF file: {path}");
    let sf = SafeFile::open(path)?;
    println!("SafeFile opened! Total tensors: {}, Metadata keys: {}", sf.tensors.len(), sf.metadata.len());
    if let Some(fmt) = sf.metadata.get("h3.quant.format") {
        println!("  h3.quant.format: {fmt}");
    }

    let prefix = "blocks.0.attn.qkv_proj";
    println!("Loading QLinear layer from GGUF: {prefix}");
    let qlinear = QLinear::load(&sf, prefix, Placement::Gpu, &dev, DType::BF16)?;
    println!("QLinear loaded successfully!");
    println!("  in_features: {}", qlinear.in_features);
    println!("  out_features: {}", qlinear.out_features);

    println!("Testing forward pass with BF16 input (M=1)...");
    let x = Tensor::randn(0.0f32, 1.0, (1, qlinear.in_features), &dev)?.to_dtype(DType::BF16)?;
    let y = qlinear.forward(&x)?;
    dev.synchronize()?;
    println!("Forward pass succeeded!");
    println!("  Output shape: {:?}", y.dims());

    // Check for NaNs
    let vals = y.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let nan_count = vals.iter().filter(|v| v.is_nan()).count();
    println!("  NaN count: {nan_count} / {}", vals.len());
    assert_eq!(nan_count, 0, "Output contains NaNs!");

    // Test with DiT token patch length (M=4096)
    println!("Testing DiT token sequence forward pass (M=4096, N={}, K={})...", qlinear.out_features, qlinear.in_features);
    let x_batch = Tensor::randn(0.0f32, 1.0, (4096, qlinear.in_features), &dev)?.to_dtype(DType::BF16)?;
    let y_batch = qlinear.forward(&x_batch)?;
    dev.synchronize()?;
    println!("  Output shape: {:?}", y_batch.dims());

    // Benchmark 20 iterations at M=4096
    println!("Benchmarking 20 forward iterations on blocks.0.attn.qkv_proj (M=4096)...");
    for _ in 0..3 {
        let _ = qlinear.forward(&x_batch)?;
    }
    dev.synchronize()?;

    let start = std::time::Instant::now();
    let iters = 20;
    for _ in 0..iters {
        let _ = qlinear.forward(&x_batch)?;
    }
    dev.synchronize()?;
    let elapsed = start.elapsed().as_secs_f64() / (iters as f64);
    let flops = 2.0 * 4096.0 * (qlinear.in_features as f64) * (qlinear.out_features as f64);
    let tops = (flops / elapsed) / 1e12;
    println!("  Average latency per QLinear forward (M=4096): {:.3} ms", elapsed * 1000.0);
    println!("  Throughput: {:.2} TOPS (Native Ampere sm_86 INT4 Tensor Cores)", tops);

    println!("=== All GGUF W4A4 Ampere INT4 Tensor Core Tests PASSED! ===");
    Ok(())
}
