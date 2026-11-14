use candle_core::{DType, Device, Tensor};
use candle_core::cuda_backend::cudarc::driver::sys;
use h3_rs::weights::{Placement, QLinear, SafeFile};

fn main() -> anyhow::Result<()> {
    println!("=== Testing Multi-Layer CUDA Graph Capture on RTX 3070 ===");
    let dev = Device::new_cuda(0)?;
    let Device::Cuda(cd) = &dev else { anyhow::bail!("CUDA device required") };
    let stream = cd.cuda_stream();

    let path = "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.safetensors";
    let sf = SafeFile::open(path)?;

    println!("Loading block 0 QLinear layers onto GPU...");
    let qkv = QLinear::load(&sf, "blocks.0.attn.qkv_proj", Placement::Gpu, &dev, DType::BF16)?;
    let out = QLinear::load(&sf, "blocks.0.attn.out_proj", Placement::Gpu, &dev, DType::BF16)?;
    let fc1 = QLinear::load(&sf, "blocks.0.mlp.fc1", Placement::Gpu, &dev, DType::BF16)?;
    let fc2 = QLinear::load(&sf, "blocks.0.mlp.fc2", Placement::Gpu, &dev, DType::BF16)?;

    let m_tokens = 4096;
    let x = Tensor::randn(0.0f32, 1.0, (m_tokens, 5376), &dev)?.to_dtype(DType::BF16)?;

    println!("Warmup forward across 4 projections...");
    let h_qkv = qkv.forward(&x)?;
    let h_out = out.forward(&x)?;
    let h_fc1 = fc1.forward(&x)?;
    let h_fc2 = fc2.forward(&x)?;
    dev.synchronize()?;
    println!("Warmup done! Shapes: qkv={:?}, out={:?}, fc1={:?}, fc2={:?}", h_qkv.dims(), h_out.dims(), h_fc1.dims(), h_fc2.dims());

    println!("Beginning CUDA Stream Capture for 4 projections...");
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)?;
    let _ = qkv.forward(&x)?;
    let _ = out.forward(&x)?;
    let _ = fc1.forward(&x)?;
    let _ = fc2.forward(&x)?;
    println!("Ending stream capture...");
    let graph = stream.end_capture(sys::CUgraphInstantiate_flags(0))?.unwrap();
    println!("CUDA Graph instantiated successfully!");

    println!("Testing 20 graph replays...");
    let t0 = std::time::Instant::now();
    for _ in 0..20 {
        graph.launch()?;
    }
    dev.synchronize()?;
    let t_graph = t0.elapsed().as_secs_f64() / 20.0;

    println!("Testing 20 standard launches...");
    let t1 = std::time::Instant::now();
    for _ in 0..20 {
        let _ = qkv.forward(&x)?;
        let _ = out.forward(&x)?;
        let _ = fc1.forward(&x)?;
        let _ = fc2.forward(&x)?;
    }
    dev.synchronize()?;
    let t_std = t1.elapsed().as_secs_f64() / 20.0;

    println!("Benchmark Results (4 linear projections per block, M=4096):");
    println!("  Standard 4-proj latency:   {:.3} ms", t_std * 1000.0);
    println!("  CUDA Graph 4-proj latency: {:.3} ms", t_graph * 1000.0);
    println!("  Speedup: {:.2}x / Launch overhead saved: {:.3} ms per block", t_std / t_graph, (t_std - t_graph) * 1000.0);
    println!("  Projected savings across 28 resident blocks: {:.2} ms per step ({:.2} s per 6-step)",
        (t_std - t_graph) * 1000.0 * 28.0,
        (t_std - t_graph) * 28.0 * 6.0
    );

    Ok(())
}
