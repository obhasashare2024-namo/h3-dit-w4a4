use candle_core::{DType, Device, Tensor};
use candle_core::cuda_backend::cudarc::driver::sys;
use h3_rs::weights::{Placement, QLinear, SafeFile};

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    let Device::Cuda(cd) = &dev else { anyhow::bail!("CUDA device required") };
    let stream = cd.cuda_stream();

    let path = "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.safetensors";
    let sf = SafeFile::open(path)?;
    let qlinear = QLinear::load(&sf, "blocks.0.attn.qkv_proj", Placement::Gpu, &dev, DType::BF16)?;

    let x = Tensor::randn(0.0f32, 1.0, (4096, qlinear.in_features), &dev)?.to_dtype(DType::BF16)?;
    let _ = qlinear.forward(&x)?;
    dev.synchronize()?;

    println!("1. Begin capture");
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)?;
    let _ = qlinear.forward(&x);
    println!("2. End capture");
    let graph = stream.end_capture(sys::CUgraphInstantiate_flags(0))?.unwrap();
    println!("3. Captured graph ok");

    println!("4. Launch 1");
    match graph.launch() {
        Ok(_) => println!("   Launch 1 OK"),
        Err(e) => println!("   Launch 1 Err: {e:?}"),
    }
    match dev.synchronize() {
        Ok(_) => println!("   Sync 1 OK"),
        Err(e) => println!("   Sync 1 Err: {e:?}"),
    }

    println!("5. Launch 2");
    match graph.launch() {
        Ok(_) => println!("   Launch 2 OK"),
        Err(e) => println!("   Launch 2 Err: {e:?}"),
    }
    match dev.synchronize() {
        Ok(_) => println!("   Sync 2 OK"),
        Err(e) => println!("   Sync 2 Err: {e:?}"),
    }

    Ok(())
}
