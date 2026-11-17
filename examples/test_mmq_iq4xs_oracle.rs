use anyhow::{bail, Result};
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::cudarc::nvrtc::compile_ptx_with_opts;
use candle_core::{DType, Device, Tensor};

const MMQ_IQ4XS_SRC: &str = r#"
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef signed char i8;

#define QK_K 256

#pragma pack(push, 1)
typedef struct {
    u16 d;
    u16 scales_h;
    u8  scales_l[4];
    u8  qs[128];
} block_iq4_xs;
#pragma pack(pop)

__constant__ i8 kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};

__device__ __forceinline__ float bf2f(u16 v) { return __uint_as_float(((u32)v) << 16); }
__device__ __forceinline__ u16 f2bf(float f) {
    u32 u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
    u += 0x7fffu + ((u >> 16) & 1u);
    return (u16)(u >> 16);
}
__device__ __forceinline__ float h2f(u16 v) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(v)); return f; }

extern "C" __global__ void mmq_iq4_xs_direct(
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
    const block_iq4_xs* w_row = ((const block_iq4_xs*)w_raw) + (size_t)n_idx * num_blocks;
    const u16* x_row = x + (size_t)m_idx * K;

    float acc = 0.0f;

    for (int b = 0; b < num_blocks; ++b) {
        const block_iq4_xs* blk = &w_row[b];
        float d = h2f(blk->d);
        u16 scales_h = blk->scales_h;
        const u8* scales_l = blk->scales_l;
        const u8* qs = blk->qs;

        #pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
            float dl = d * (float)(ls - 32);

            int k_base = b * QK_K + 32 * ib;
            const u8* sub_qs = qs + 16 * ib;

            float sub_dot = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                u8 q = sub_qs[j];
                float v0 = (float)kvalues_iq4nl[q & 0xF];
                float v1 = (float)kvalues_iq4nl[q >> 4];

                float x0 = bf2f(x_row[k_base + j]);
                float x1 = bf2f(x_row[k_base + 16 + j]);

                sub_dot += x0 * v0 + x1 * v1;
            }

            acc += dl * sub_dot;
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

// CPU reference dequantization using official GGML IQ4_XS arithmetic
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
];

fn f16_to_f32(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}

fn dequant_iq4_xs_cpu(raw: &[u8], num_blocks: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; num_blocks * 256];
    for b in 0..num_blocks {
        let base = b * 136;
        let d_raw = u16::from_le_bytes([raw[base], raw[base + 1]]);
        let d = f16_to_f32(d_raw);
        let scales_h = u16::from_le_bytes([raw[base + 2], raw[base + 3]]);
        let scales_l = &raw[base + 4..base + 8];
        let qs = &raw[base + 8..base + 136];

        for ib in 0..8 {
            let ls = (((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32)
                | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
            let dl = d * (ls - 32) as f32;

            let out_base = b * 256 + 32 * ib;
            let sub_qs = &qs[16 * ib..16 * (ib + 1)];

            for j in 0..16 {
                let q = sub_qs[j];
                let v0 = KVALUES_IQ4NL[(q & 0xF) as usize] as f32;
                let v1 = KVALUES_IQ4NL[(q >> 4) as usize] as f32;

                out[out_base + j] = dl * v0;
                out[out_base + 16 + j] = dl * v1;
            }
        }
    }
    out
}

fn main() -> Result<()> {
    println!("========================================================");
    println!("  DEF-043: MMQ IQ4_XS Direct Non-Linear Codebook Dot Test ");
    println!("========================================================");

    let dev = Device::new_cuda(0)?;
    let in_features = 5376usize;
    let out_features = 256usize;
    let m = 32usize;

    println!("Shape: Tokens M={}, In K={}, Out N={}", m, in_features, out_features);

    // 1. Generate valid IQ4_XS raw block buffer (136 bytes per 256 weights)
    let num_blocks = (out_features * in_features) / 256;
    let raw_bytes_len = num_blocks * 136;
    let mut raw = vec![0x55u8; raw_bytes_len];
    for b in 0..num_blocks {
        let base = b * 136;
        // d: 2 bytes fp16 (0.05 in f16 = 0x2A66)
        raw[base + 0] = 0x66;
        raw[base + 1] = 0x2A;
        // scales_h: 2 bytes
        raw[base + 2] = ((b * 7 + 0x55) & 0xFF) as u8;
        raw[base + 3] = ((b * 11 + 0xAA) & 0xFF) as u8;
        // scales_l: 4 bytes
        for s in 0..4 {
            raw[base + 4 + s] = (((s as u8 * 17) ^ 0x33) & 0xFF) as u8;
        }
        // qs: 128 bytes
        for q in 0..128 {
            raw[base + 8 + q] = (((q as u8 * 29) ^ 0x9B) & 0xFF) as u8;
        }
    }
    let raw_t = Tensor::from_vec(raw.clone(), (raw_bytes_len,), &dev)?;

    // 2. Generate random BF16 activations
    let x_bf16 = Tensor::randn(0.0f32, 1.0, (m, in_features), &dev)?.to_dtype(DType::BF16)?;

    // 3. Ground Truth: CPU Dequantize -> GPU cuBLAS matmul
    println!("Dequantizing IQ4_XS blocks via CPU official GGML codebook...");
    let t0 = std::time::Instant::now();
    let w_f32_vec = dequant_iq4_xs_cpu(&raw, num_blocks);
    let w_bf16 = Tensor::from_vec(w_f32_vec, (out_features, in_features), &Device::Cpu)?
        .to_dtype(DType::BF16)?
        .to_device(&dev)?;
    let y_ref = x_bf16.matmul(&w_bf16.t()?)?;
    dev.synchronize()?;
    let ref_elapsed = t0.elapsed();
    println!("Ground Truth (CPU Dequant + GPU matmul): {:.3} ms", ref_elapsed.as_secs_f64() * 1000.0);

    // 4. Compile MMQ kernel via NVRTC
    let opts = candle_core::cuda_backend::cudarc::nvrtc::CompileOptions {
        use_fast_math: Some(false),
        options: vec!["--std=c++17".into(), "--gpu-architecture=compute_86".into()],
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(MMQ_IQ4XS_SRC, opts)?;
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
    let func = cuda_dev.get_or_load_custom_func("mmq_iq4_xs_direct", "mmq_iq4xs_module", &ptx_src)?;

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
    println!("MMQ Direct (Raw IQ4_XS block dot product): {:.3} ms", mmq_elapsed.as_secs_f64() * 1000.0);

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

    println!("\n🎉 DEF-043 MMQ IQ4_XS DIRECT DOT PRODUCT VERIFIED 100%!");
    Ok(())
}
