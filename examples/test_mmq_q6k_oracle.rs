use anyhow::{bail, Result};
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::cudarc::nvrtc::compile_ptx_with_opts;
use candle_core::{DType, Device, Tensor};
use h3_rs::kernels;
use h3_rs::weights::GgmlType;

const MMQ_Q6K_SRC: &str = r#"
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;

#define QK_K 256

#pragma pack(push, 1)
typedef struct {
    u8  ql[QK_K / 2];
    u8  qh[QK_K / 4];
    signed char scales[QK_K / 16];
    u16 d;
} block_q6_K;
#pragma pack(pop)

__device__ __forceinline__ float bf2f(u16 v) { return __uint_as_float(((u32)v) << 16); }
__device__ __forceinline__ u16 f2bf(float f) {
    u32 u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
    u += 0x7fffu + ((u >> 16) & 1u);
    return (u16)(u >> 16);
}
__device__ __forceinline__ float h2f(u16 v) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(v)); return f; }

extern "C" __global__ void mmq_q6_k_direct(
    const u16* __restrict__ x,
    const void* __restrict__ w_raw,
    u16* __restrict__ y,
    int M,
    int N,
    int K
) {
    int m_idx = blockIdx.y;
    int n_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (m_idx >= M || n_idx >= N) return;

    int num_blocks = K / QK_K;
    const block_q6_K* w_row = ((const block_q6_K*)w_raw) + (size_t)n_idx * num_blocks;
    const u16* x_row = x + (size_t)m_idx * K;

    float acc = 0.0f;

    for (int b = 0; b < num_blocks; ++b) {
        const block_q6_K* blk = &w_row[b];
        float d = h2f(blk->d);

        #pragma unroll
        for (int ip = 0; ip < 2; ++ip) {
            const u8* ql_base = blk->ql + 64 * ip;
            const u8* qh_base = blk->qh + 32 * ip;
            const signed char* sc_base = blk->scales + 8 * ip;
            int k_half = b * QK_K + 128 * ip;

            #pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                const signed char* sc = sc_base + sub;
                float s0 = d * (float)sc[0];
                float s2 = d * (float)sc[2];
                float s4 = d * (float)sc[4];
                float s6 = d * (float)sc[6];

                #pragma unroll
                for (int l = 0; l < 16; ++l) {
                    int il = sub * 16 + l;
                    u8 ql_0 = ql_base[il];
                    u8 ql_32 = ql_base[32 + il];
                    u8 qh = qh_base[il];

                    float w0 = (float)((signed char)((ql_0 & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
                    float w1 = (float)((signed char)((ql_32 & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
                    float w2 = (float)((signed char)((ql_0 >> 4) | (((qh >> 4) & 3) << 4)) - 32);
                    float w3 = (float)((signed char)((ql_32 >> 4) | (((qh >> 6) & 3) << 4)) - 32);

                    float x0 = bf2f(x_row[k_half + il + 0]);
                    float x1 = bf2f(x_row[k_half + il + 32]);
                    float x2 = bf2f(x_row[k_half + il + 64]);
                    float x3 = bf2f(x_row[k_half + il + 96]);

                    acc += s0 * (w0 * x0) + s2 * (w1 * x1) + s4 * (w2 * x2) + s6 * (w3 * x3);
                }
            }
        }
    }

    y[(size_t)m_idx * N + n_idx] = f2bf(acc);
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
    println!("  DEF-037: MMQ Q6_K Direct Quantized Dot Product Test   ");
    println!("========================================================");

    let dev = Device::new_cuda(0)?;
    let in_features = 5376usize;
    let out_features = 256usize;
    let m = 32usize;

    println!("Shape: Tokens M={}, In K={}, Out N={}", m, in_features, out_features);

    // 1. Generate valid Q6_K raw block buffer (210 bytes per 256 weights)
    let num_blocks = (out_features * in_features) / 256;
    let raw_bytes_len = num_blocks * 210;
    let mut raw = vec![0x55u8; raw_bytes_len];
    for b in 0..num_blocks {
        let base = b * 210;
        // ql: 128 bytes
        for q in 0..128 {
            raw[base + q] = ((q as u8 * 17) ^ 0x93) & 0xFF;
        }
        // qh: 64 bytes
        for q in 0..64 {
            raw[base + 128 + q] = ((q as u8 * 31) ^ 0x47) & 0xFF;
        }
        // scales: 16 bytes
        for s in 0..16 {
            raw[base + 192 + s] = (((s as i8 * 7) ^ 0x23) as u8) & 0xFF;
        }
        // d: 2 bytes fp16 (0.05 in f16 = 0x2A66)
        raw[base + 208] = 0x66;
        raw[base + 209] = 0x2A;
    }
    let raw_t = Tensor::from_vec(raw.clone(), (raw_bytes_len,), &dev)?;

    // 2. Generate random BF16 activations
    let x_bf16 = Tensor::randn(0.0f32, 1.0, (m, in_features), &dev)?.to_dtype(DType::BF16)?;

    // 3. Ground Truth: DequantArena + cuBLAS matmul
    dev.synchronize()?;
    let t0 = std::time::Instant::now();
    let w_bf16 = kernels::dequant_gguf(&raw_t, GgmlType::Q6K, out_features, in_features, DType::BF16)?;
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
    let ptx = compile_ptx_with_opts(MMQ_Q6K_SRC, opts)?;
    let mut ptx_src = ptx.to_src();

    // Apply PTX version compatibility fix
    for v in ["9.4", "9.3", "9.2", "9.1", "9.0", "8.9", "8.8", "8.7", "8.6"] {
        if ptx_src.contains(&format!(".version {}", v)) {
            ptx_src = ptx_src.replace(&format!(".version {}", v), ".version 8.5");
            break;
        }
    }

    let cuda_dev = match &dev {
        Device::Cuda(c) => c.clone(),
        _ => bail!("CUDA required"),
    };
    let func = cuda_dev.get_or_load_custom_func("mmq_q6_k_direct", "mmq_q6k_module", &ptx_src)?;

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
    println!("MMQ Direct (Raw Q6_K block dot product): {:.3} ms", mmq_elapsed.as_secs_f64() * 1000.0);

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

    println!("\n🎉 DEF-037 MMQ Q6_K DIRECT DOT PRODUCT VERIFIED 100%!");
    Ok(())
}