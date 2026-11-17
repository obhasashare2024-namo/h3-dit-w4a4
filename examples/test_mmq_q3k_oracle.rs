use anyhow::{bail, Result};
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::cudarc::nvrtc::compile_ptx_with_opts;
use candle_core::{DType, Device, Tensor};
use h3_rs::kernels;
use h3_rs::weights::GgmlType;

const MMQ_SRC: &str = r#"
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;

#define QK_K 256

typedef struct {
    u8 hmask[32];
    u8 qs[64];
    u8 scales[12];
    u16 d;
} block_q3_K;

__device__ __forceinline__ float bf2f(u16 v) { return __uint_as_float(((u32)v) << 16); }
__device__ __forceinline__ u16 f2bf(float f) {
    u32 u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
    u += 0x7fffu + ((u >> 16) & 1u);
    return (u16)(u >> 16);
}
__device__ __forceinline__ float h2f(u16 v) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(v)); return f; }

extern "C" __global__ void mmq_q3_k_direct(
    const u16* __restrict__ x,
    const void* __restrict__ w_raw,
    u16* __restrict__ y,
    int M,
    int N,
    int K
) {
    int m = blockIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    int num_blocks = K / QK_K;
    const block_q3_K* w_row = ((const block_q3_K*)w_raw) + (size_t)n * num_blocks;
    const u16* x_row = x + (size_t)m * K;

    float acc = 0.0f;

    for (int b = 0; b < num_blocks; ++b) {
        const block_q3_K* blk = &w_row[b];
        float d_all = h2f(blk->d);

        #pragma unroll
        for (int s = 0; s < 16; ++s) {
            int n_idx = s / 8;
            int r = s % 8;
            int j = r / 2;
            int is0 = r % 2;

            u8 mask = 1 << (4 * n_idx + j);
            int shift = 2 * j;
            int is = 8 * n_idx + 2 * j + is0;

            signed char us = is < 4 ? (blk->scales[is - 0] & 0xF) | (((blk->scales[is + 8] >> 0) & 3) << 4) :
                             is < 8 ? (blk->scales[is - 0] & 0xF) | (((blk->scales[is + 4] >> 2) & 3) << 4) :
                             is < 12 ? (blk->scales[is - 8] >> 4) | (((blk->scales[is + 0] >> 4) & 3) << 4) :
                                       (blk->scales[is - 8] >> 4) | (((blk->scales[is - 4] >> 6) & 3) << 4);

            float dl = d_all * (float)(us - 32);

            const u8* q = blk->qs + 32 * n_idx;
            const u8* hm = blk->hmask;
            int l_offset = 16 * is0;

            int k_sub_base = b * QK_K + 128 * n_idx + 32 * j + l_offset;

            float sub_dot = 0.0f;
            #pragma unroll
            for (int l = 0; l < 16; ++l) {
                int l_curr = l_offset + l;
                float w_val = (float)((signed char)((q[l_curr] >> shift) & 3) - ((hm[l_curr] & mask) ? 0 : 4));
                float x_val = bf2f(x_row[k_sub_base + l]);
                sub_dot += x_val * w_val;
            }

            acc += dl * sub_dot;
        }
    }

    y[(size_t)m * N + n] = f2bf(acc);
}
"#;

fn tptr(t: &Tensor) -> Result<u64> {
    use candle_core::cuda_backend::CudaStorageSlice as Sl;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    let (st, l) = t.storage_and_layout();
    match &*st {
        candle_core::Storage::Cuda(c) => {
            let stream = c.device.cuda_stream();
            let off = l.start_offset();
            Ok(match &c.slice {
                Sl::U8(s) => s.device_ptr(&stream).0 + off as u64,
                Sl::F32(s) => s.device_ptr(&stream).0 + (off * 4) as u64,
                Sl::BF16(s) => s.device_ptr(&stream).0 + (off * 2) as u64,
                Sl::F16(s) => s.device_ptr(&stream).0 + (off * 2) as u64,
                _ => bail!("unsupported storage"),
            })
        }
        _ => bail!("expected cuda tensor"),
    }
}

fn main() -> Result<()> {
    println!("========================================================");
    println!("  DEF-027: MMQ Q3_K Direct Quantized Dot Product Test   ");
    println!("========================================================");

    let dev = Device::new_cuda(0)?;
    let in_features = 5376usize;
    let out_features = 256usize;
    let m = 32usize;

    println!("Shape: Tokens M={}, In K={}, Out N={}", m, in_features, out_features);

    // 1. Generate valid Q3_K raw block buffer with diverse scales and weights
    let num_blocks = (out_features * in_features) / 256;
    let raw_bytes_len = num_blocks * 110;
    let mut raw = vec![0x55u8; raw_bytes_len];
    for b in 0..num_blocks {
        let base = b * 110;
        raw[base + 108] = 0x66;
        raw[base + 109] = 0x2A; // 0.05 in f16
        for s in 0..12 {
            raw[base + 96 + s] = (s as u8 * 17 + b as u8 * 7) ^ 0xAA;
        }
    }
    let raw_t = Tensor::from_vec(raw.clone(), (raw_bytes_len,), &dev)?;

    // 2. Generate random BF16 activations
    let x_bf16 = Tensor::randn(0.0f32, 1.0, (m, in_features), &dev)?.to_dtype(DType::BF16)?;

    // 3. Ground Truth: DequantArena + cuBLAS matmul
    dev.synchronize()?;
    let t0 = std::time::Instant::now();
    let w_bf16 = kernels::dequant_gguf(&raw_t, GgmlType::Q3K, out_features, in_features, DType::BF16)?;
    let y_ref = x_bf16.matmul(&w_bf16.t()?)?;
    dev.synchronize()?;
    let ref_elapsed = t0.elapsed();
    println!("Ground Truth (DequantArena + cuBLAS): {:.3} ms", ref_elapsed.as_secs_f64() * 1000.0);

    // 4. Compile MMQ kernel via NVRTC
    let opts = candle_core::cuda_backend::cudarc::nvrtc::CompileOptions {
        use_fast_math: Some(false),
        options: vec!["--std=c++17".into(), "--gpu-architecture=compute_86".into()],
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(MMQ_SRC, opts)?;
    let mut ptx_src = ptx.to_src();
    println!("Generated PTX first 3 lines:\n{}", ptx_src.lines().take(3).collect::<Vec<_>>().join("\n"));

    // Apply PTX version compatibility fix
    for v in ["9.4", "9.3", "9.2", "9.1", "9.0", "8.9", "8.8", "8.7", "8.6"] {
        if ptx_src.contains(&format!(".version {}", v)) {
            println!("Replacing .version {} with .version 8.5", v);
            ptx_src = ptx_src.replace(&format!(".version {}", v), ".version 8.5");
            break;
        }
    }

    let cuda_dev = match &dev {
        Device::Cuda(c) => c.clone(),
        _ => bail!("CUDA required"),
    };
    let func = cuda_dev.get_or_load_custom_func("mmq_q3_k_direct", "mmq_module", &ptx_src)?;

    // 5. Allocate MMQ output tensor and launch directly on raw bytes (ZERO weight dequantization!)
    let y_mmq = Tensor::zeros((m, out_features), DType::BF16, &dev)?;

    let xp = tptr(&x_bf16)?;
    let wp = tptr(&raw_t)?;
    let yp = tptr(&y_mmq)?;

    let (mi, ni, ki) = (m as i32, out_features as i32, in_features as i32);
    let mut bld = func.builder();
    bld.arg(&xp).arg(&wp).arg(&yp).arg(&mi).arg(&ni).arg(&ki);

    let block_dim = 128u32;
    let grid_x = (out_features as u32).div_ceil(block_dim);
    let grid_y = m as u32;

    dev.synchronize()?;
    let t1 = std::time::Instant::now();
    unsafe {
        bld.launch(LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    dev.synchronize()?;
    let mmq_elapsed = t1.elapsed();
    println!("MMQ Direct (Raw block dot product):   {:.3} ms", mmq_elapsed.as_secs_f64() * 1000.0);

    // 6. Compare Outputs: Bitwise & Precision Analysis
    let y_ref_f32 = y_ref.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let y_mmq_f32 = y_mmq.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

    let mut max_abs_diff = 0.0f32;
    let mut dot_product = 0.0f64;
    let mut norm_ref = 0.0f64;
    let mut norm_mmq = 0.0f64;
    let mut has_nan = false;
    let mut has_inf = false;

    for i in 0..y_ref_f32.len() {
        let r = y_ref_f32[i];
        let q = y_mmq_f32[i];

        if r.is_nan() || q.is_nan() { has_nan = true; }
        if r.is_infinite() || q.is_infinite() { has_inf = true; }

        let diff = (r - q).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }

        dot_product += (r as f64) * (q as f64);
        norm_ref += (r as f64) * (r as f64);
        norm_mmq += (q as f64) * (q as f64);
    }

    let cosine = dot_product / (norm_ref.sqrt() * norm_mmq.sqrt());
    let rel_diff = max_abs_diff / (y_ref_f32.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-5));

    println!("\n=== PRECISION & QUALITY GATES ===");
    println!("  Total output elements: {}", y_ref_f32.len());
    println!("  Max Absolute Diff:     {:.6}", max_abs_diff);
    println!("  Relative Diff:         {:.6} (Gate: < 0.05)", rel_diff);
    println!("  Cosine Similarity:     {:.6} (Gate: > 0.999)", cosine);
    println!("  Has NaN:               {} (Gate: false)", has_nan);
    println!("  Has Inf:               {} (Gate: false)", has_inf);

    assert!(!has_nan, "NaN in MMQ output!");
    assert!(!has_inf, "Inf in MMQ output!");
    assert!(cosine > 0.999, "Cosine similarity failed gate!");
    assert!(rel_diff < 0.05, "Relative difference failed gate!");

    println!("\n🎉 DEF-027 MMQ Q3_K DIRECT DOT PRODUCT VERIFIED 100%!");
    Ok(())
}
