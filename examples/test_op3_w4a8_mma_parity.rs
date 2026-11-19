use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use h3_rs::weights::{Placement, QLinear, SafeFile};
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<()> {
    println!("========================================================");
    println!("  Rank 2: OP-3 W4A8 INT8 MMA vs cuBLASLt Parity & Bench ");
    println!("========================================================");

    let dev = Device::new_cuda(0)?;
    let model_path = "/mnt/2tbnvS/models/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors";

    println!("Loading W4A8 fc1 layer...");
    let file = Arc::new(SafeFile::open(model_path)?);
    let mut fc1 = QLinear::load(&file, "blocks.0.mlp.fc1", Placement::Disk, &dev, DType::BF16)?;
    fc1.weight.promote(&dev)?;

    // Create realistic activation matching seq_len=8439
    let m = 8439usize;
    let k = 5376usize;
    let x_bf16 = Tensor::randn(0.0f32, 1.0, (m, k), &dev)?.to_dtype(DType::BF16)?;

    println!("Input shape: {:?}, dtype: {:?}", x_bf16.dims(), x_bf16.dtype());

    // Prepare W4A8 weights into INT8 tensor
    let raw = fc1.fetch(&dev)?;
    let scale = fc1.scale.as_ref().unwrap().get(&dev)?;
    let codebook = match &fc1.fmt {
        h3_rs::weights::QFormat::AsymW4A8 { codebook, .. } => codebook.get(&dev)?,
        _ => anyhow::bail!("Expected AsymW4A8 format"),
    };
    let s_rel = raw.bs.as_ref().unwrap();
    let w_i8 = h3_rs::kernels::dequant_w4a8_to_i8(&raw.w, s_rel, &codebook, fc1.in_features)?;

    println!("Materialized INT8 Weight shape: {:?}, scale shape: {:?}", w_i8.dims(), scale.dims());

    // 1. Direct cuBLASLt Kernel
    println!("\n--- 1. Testing cuBLASLt Path (kernels::int8_linear_ex) ---");
    let iters = 20;
    // Warmup
    let _ = h3_rs::kernels::int8_linear_ex(&x_bf16, &w_i8, &scale, fc1.bias.as_ref(), DType::BF16, false, false)?;
    dev.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = h3_rs::kernels::int8_linear_ex(&x_bf16, &w_i8, &scale, fc1.bias.as_ref(), DType::BF16, false, false)?;
        dev.synchronize()?;
    }
    let cublas_ms = t0.elapsed().as_secs_f64() / iters as f64 * 1000.0;
    let y_cublas = h3_rs::kernels::int8_linear_ex(&x_bf16, &w_i8, &scale, fc1.bias.as_ref(), DType::BF16, false, false)?;
    dev.synchronize()?;
    println!("cuBLASLt Latency: {:.2} ms per GEMM", cublas_ms);

    // 2. Direct Ampere m16n8k32 MMA Kernel
    println!("\n--- 2. Testing Ampere m16n8k32 MMA Path (attn8::gemm_s8) ---");
    // Warmup
    let _ = h3_rs::attn8::gemm_s8(&x_bf16, &w_i8, &scale, fc1.bias.as_ref(), false, false)?;
    dev.synchronize()?;
    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = h3_rs::attn8::gemm_s8(&x_bf16, &w_i8, &scale, fc1.bias.as_ref(), false, false)?;
        dev.synchronize()?;
    }
    let mma_ms = t1.elapsed().as_secs_f64() / iters as f64 * 1000.0;
    let y_mma = h3_rs::attn8::gemm_s8(&x_bf16, &w_i8, &scale, fc1.bias.as_ref(), false, false)?;
    dev.synchronize()?;
    println!("Ampere MMA Latency: {:.2} ms per GEMM", mma_ms);

    let speedup = cublas_ms / mma_ms;
    println!("Speedup (cuBLASLt / MMA): {:.2}x", speedup);

    // 3. Bitwise & Numerical Parity Analysis
    println!("\n--- 3. Numerical Parity Analysis ---");
    let diff = (&y_cublas - &y_mma)?.to_dtype(DType::F32)?;
    let abs_diff = diff.abs()?;
    let max_diff = abs_diff.flatten_all()?.max(0)?.to_vec0::<f32>()?;
    let mean_diff = (abs_diff.sum_all()?.to_vec0::<f32>()? as f64) / (diff.elem_count() as f64);

    let norm_cublas = (y_cublas.to_dtype(DType::F32)?.sqr()?.sum_all()?.to_vec0::<f32>()? as f64).sqrt();
    let norm_diff = (diff.sqr()?.sum_all()?.to_vec0::<f32>()? as f64).sqrt();
    let rel_l2 = norm_diff / norm_cublas;

    println!("Max Absolute Diff : {:.6e}", max_diff);
    println!("Mean Absolute Diff: {:.6e}", mean_diff);
    println!("Relative L2 Error : {:.6e}", rel_l2);

    let ops = 2.0 * (m as f64) * (k as f64) * (fc1.out_features as f64);
    let cublas_tops = (ops / (cublas_ms * 1e-3)) / 1e12;
    let mma_tops = (ops / (mma_ms * 1e-3)) / 1e12;
    println!("cuBLASLt Throughput: {:.2} TOPS", cublas_tops);
    println!("Ampere MMA Throughput: {:.2} TOPS", mma_tops);

    assert!(rel_l2 < 0.05, "Relative L2 error must be < 5% for INT8 quant parity");
    println!("\n🎉 OP-3 W4A8 INT8 MMA Parity Verification PASSED 100%!");
    Ok(())
}
