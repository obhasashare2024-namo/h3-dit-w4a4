// ============================================================================
// File: w4a4_ampere_kernels.cu
// Target Architecture: NVIDIA Ampere (SM80 / SM86, e.g., A100, RTX 3090)
// Compiler Compatibility: Pure NVRTC (Zero external header dependencies) & NVCC
// ============================================================================

#if defined(__CUDACC_RTC__)
typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef short              int16_t;
typedef unsigned short     uint16_t;
typedef int                int32_t;
typedef unsigned int       uint32_t;
typedef long long          int64_t;
typedef unsigned long long uint64_t;
#else
#include <stdint.h>
#endif

#define WARP_SIZE 32
#define FULL_WARP_MASK 0xffffffffu

// ============================================================================
// Section 1: Inline PTX Type Conversions & Dead-Zone-Free Quantization Helpers
// ============================================================================

__device__ __forceinline__ float half_to_float(uint16_t h) {
    float f;
    asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
    return f;
}

__device__ __forceinline__ uint16_t float_to_half(float f) {
    uint16_t h;
    asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(h) : "f"(f));
    return h;
}

__device__ __forceinline__ uint32_t float2_to_half2(float f0, float f1) {
    uint16_t h0 = float_to_half(f0);
    uint16_t h1 = float_to_half(f1);
    return (static_cast<uint32_t>(h1) << 16) | static_cast<uint32_t>(h0);
}

// Dead-zone-free symmetric INT4 quantization:
// Uses round-to-nearest-even (cvt.rni) instead of truncation to eliminate
// the zero dead-zone, clamping to symmetric [-7, 7] (or [-8, 7]).
__device__ __forceinline__ int32_t quant_s4_deadzone_free(float val, float inv_scale) {
    float scaled = val * inv_scale;
    int32_t q;
    asm volatile("cvt.rni.s32.f32 %0, %1;\n" : "=r"(q) : "f"(scaled));
    q = (q < -7) ? -7 : ((q > 7) ? 7 : q);
    return q & 0xF;
}

// ============================================================================
// Section 2: Swizzle<3, 4, 3> Zero Bank-Conflict Shared Memory Layout
// ============================================================================
// Row stride = 256 INT4 elements = 128 Bytes = 8 x 16B vectors.
// Byte address bits:
//   bits [3:0] -> 16B vector internal offset (M = 4)
//   bits [6:4] -> 16B column index (0..7)    (B = 3)
//   bits [9:7] -> Row index mod 8  (0..7)    (S = 3)
// Swizzle<3,4,3> XORs bits [6:4] with bits [9:7].

__device__ __forceinline__ uint32_t swizzle_3_4_3_byte(uint32_t byte_offset) {
    return byte_offset ^ ((byte_offset >> 3) & 0x70u);
}

__device__ __forceinline__ uint32_t smem_offset_swizzle_3_4_3(uint32_t row, uint32_t col_16B) {
    uint32_t swizzled_col_16B = col_16B ^ (row & 0x7u);
    return (row << 7) | (swizzled_col_16B << 4);
}

// ============================================================================
// Section 3: Ampere cp.async, ldmatrix, and mma.sync.m16n8k64.s4 Primitives
// ============================================================================

__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_cg_16B_pred(
    uint32_t smem_addr,
    const void* gmem_ptr,
    bool pred
) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
        "  @!p st.shared.v4.u32 [%0], {0, 0, 0, 0};\n"
        "}\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr), "r"(static_cast<int32_t>(pred))
        : "memory"
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" :: : "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N) : "memory");
}

__device__ __forceinline__ void ldmatrix_m8n8_x4_b16(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    uint32_t smem_addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_addr)
    );
}

__device__ __forceinline__ void mma_m16n8k64_s4(
    int32_t& d0, int32_t& d1, int32_t& d2, int32_t& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1)
    );
}

// ============================================================================
// Section 4: 8-Stage Orthonormal H_256 + Dead-Zone-Free Dynamic Quantization
// ============================================================================
// - In-register H_8 butterfly: strides 1, 2, 4
// - Warp-shuffle H_32 butterfly: XOR masks 1, 2, 4, 8, 16
// - Orthonormal scaling: 0.0625f (1 / sqrt(256))
// - Dead-zone-free dynamic quantization to packed INT4 + per-group FP32/FP16 scale

__device__ __forceinline__ void fwht_h256_orthonormal_warp(float v[8], int lane_id) {
    // Stage 1..3: In-register H_8 (strides 1, 2, 4)
    #pragma unroll
    for (int i = 0; i < 8; i += 2) {
        float a = v[i];
        float b = v[i + 1];
        v[i]     = a + b;
        v[i + 1] = a - b;
    }

    #pragma unroll
    for (int i = 0; i < 8; i += 4) {
        float a0 = v[i],     b0 = v[i + 2];
        float a1 = v[i + 1], b1 = v[i + 3];
        v[i]     = a0 + b0;  v[i + 2] = a0 - b0;
        v[i + 1] = a1 + b1;  v[i + 3] = a1 - b1;
    }

    {
        float a0 = v[0], b0 = v[4];
        float a1 = v[1], b1 = v[5];
        float a2 = v[2], b2 = v[6];
        float a3 = v[3], b3 = v[7];
        v[0] = a0 + b0;  v[4] = a0 - b0;
        v[1] = a1 + b1;  v[5] = a1 - b1;
        v[2] = a2 + b2;  v[6] = a2 - b2;
        v[3] = a3 + b3;  v[7] = a3 - b3;
    }

    // Stage 4..8: Warp-level H_32 (XOR masks 1, 2, 4, 8, 16)
    #pragma unroll
    for (int mask = 1; mask <= 16; mask <<= 1) {
        bool neg = (lane_id & mask) != 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float partner = __shfl_xor_sync(FULL_WARP_MASK, v[i], mask);
            v[i] = neg ? (partner - v[i]) : (v[i] + partner);
        }
    }

    // Orthonormal normalization factor: 1.0f / sqrt(256.0f) = 0.0625f
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] *= 0.0625f;
    }
}

extern "C" __global__ void w4a4_h256_dynamic_quant_kernel(
    const uint4* __restrict__ input_f16,     // [num_groups, 32] (256 FP16 per group)
    uint32_t* __restrict__ output_packed_s4, // [num_groups, 32] (256 INT4 per group)
    float* __restrict__ output_scales,       // [num_groups]
    int num_groups
) {
    int warps_per_block = blockDim.x >> 5;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int group_idx = blockIdx.x * warps_per_block + warp_id;

    if (group_idx >= num_groups) return;

    // Coalesced 128-bit load: 8 FP16 elements per thread = 256 FP16 per warp
    uint4 raw = input_f16[group_idx * WARP_SIZE + lane_id];
    uint16_t h[8];
    h[0] = static_cast<uint16_t>(raw.x & 0xFFFFu);
    h[1] = static_cast<uint16_t>(raw.x >> 16);
    h[2] = static_cast<uint16_t>(raw.y & 0xFFFFu);
    h[3] = static_cast<uint16_t>(raw.y >> 16);
    h[4] = static_cast<uint16_t>(raw.z & 0xFFFFu);
    h[5] = static_cast<uint16_t>(raw.z >> 16);
    h[6] = static_cast<uint16_t>(raw.w & 0xFFFFu);
    h[7] = static_cast<uint16_t>(raw.w >> 16);

    float v[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = half_to_float(h[i]);
    }

    // Execute 8-stage orthonormal H_256
    fwht_h256_orthonormal_warp(v, lane_id);

    // Dynamic group amax reduction across 256 elements
    float local_amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float abs_v = fabsf(v[i]);
        local_amax = (abs_v > local_amax) ? abs_v : local_amax;
    }

    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        float other_amax = __shfl_xor_sync(FULL_WARP_MASK, local_amax, mask);
        local_amax = (other_amax > local_amax) ? other_amax : local_amax;
    }

    float scale = (local_amax > 1e-8f) ? (local_amax * (1.0f / 7.0f)) : 1e-8f;
    float inv_scale = (local_amax > 1e-8f) ? (7.0f / local_amax) : 0.0f;

    // Pack 8 dead-zone-free INT4 values into one uint32_t
    uint32_t packed = 0u;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t q = static_cast<uint32_t>(quant_s4_deadzone_free(v[i], inv_scale));
        packed |= (q << (i * 4));
    }

    output_packed_s4[group_idx * WARP_SIZE + lane_id] = packed;
    if (lane_id == 0) {
        output_scales[group_idx] = scale;
    }
}

// ============================================================================
// Section 5: Double-Buffered cp.async + Register Double-Buffered W4A4 GEMM
// ============================================================================
// Tile Configuration:
//   CTA Tile : BLK_M = 64, BLK_N = 64, BLK_K = 256 (128 Bytes along K)
//   Warps    : 4 warps (128 threads), arranged as 2x2 warps
//   Warp Tile: WARP_M = 32, WARP_N = 32, WARP_K = 64 per MMA step (4 steps/tile)
//   Smem     : 2 Stages x (64x128B for A + 64x128B for B) = 32 KB (Swizzle<3,4,3>)
//   Scale    : Supports per-group (G=256) scales [M, K/256] and [N, K/256]

extern "C" __global__ __launch_bounds__(128, 2)
void w4a4_ampere_gemm_swizzle_pipeline_kernel(
    const uint8_t* __restrict__ A_packed, // [M, K / 2] row-major packed s4
    const uint8_t* __restrict__ B_packed, // [N, K / 2] col-major (row-major on N) packed s4
    const float* __restrict__ scale_A,    // [M, K / 256] per-256-group scale for A
    const float* __restrict__ scale_B,    // [N, K / 256] per-256-group scale for B
    uint16_t* __restrict__ C_f16,         // [M, N] FP16 output
    int M,
    int N,
    int K                                 // Must be a multiple of 256
) {
    // 32 KB static shared memory: 2 stages * (8192B A + 8192B B)
    __shared__ __align__(128) uint8_t smem_buf[2][2][64 * 128];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int warp_m_idx = warp_id >> 1; // 0..1 (WARP_M = 32)
    const int warp_n_idx = warp_id & 1;  // 0..1 (WARP_N = 32)

    const int cta_m_base = blockIdx.y * 64;
    const int cta_n_base = blockIdx.x * 64;

    const int num_k_tiles = K >> 8; // K / 256
    const int k_bytes     = K >> 1; // K / 2 bytes per row

    // ------------------------------------------------------------------------
    // Global-to-Shared cp.async Mapping (128 threads):
    // Each stage of A or B is 64 rows x 128 bytes = 8192 bytes = 512 x 16B.
    // 128 threads issue 4 x 16B loads for A and 4 x 16B loads for B.
    // Within each load step (0..3):
    //   128 threads cover 16 rows x 8 cols_16B (row = tid >> 3, col_16B = tid & 7)
    // ------------------------------------------------------------------------
    const int load_row_in_chunk = tid >> 3; // 0..15
    const int load_col_16B      = tid & 7;  // 0..7

    // ------------------------------------------------------------------------
    // Shared-to-Register ldmatrix.x4 Mapping (per warp):
    // For A (WARP_M = 32 = 2 x m16 tiles):
    //   lanes 0..7   -> rows 0..7,  col_16B = 2*k_step + 0
    //   lanes 8..15  -> rows 8..15, col_16B = 2*k_step + 0
    //   lanes 16..23 -> rows 0..7,  col_16B = 2*k_step + 1
    //   lanes 24..31 -> rows 8..15, col_16B = 2*k_step + 1
    // For B (WARP_N = 32 = 2 x n16 pairs = 4 x n8 tiles):
    //   lanes 0..7   -> rows 0..7,  col_16B = 2*k_step + 0 (n8 tile 0, k0)
    //   lanes 8..15  -> rows 0..7,  col_16B = 2*k_step + 1 (n8 tile 0, k1)
    //   lanes 16..23 -> rows 8..15, col_16B = 2*k_step + 0 (n8 tile 1, k0)
    //   lanes 24..31 -> rows 8..15, col_16B = 2*k_step + 1 (n8 tile 1, k1)
    // ------------------------------------------------------------------------
    const int a_lane_row_offset = lane_id & 15;
    const int a_lane_col_offset = lane_id >> 4;

    const int b_lane_row_offset = ((lane_id >> 4) << 3) | (lane_id & 7);
    const int b_lane_col_offset = (lane_id >> 3) & 1;

    // Double-buffered MMA fragments in registers:
    // frag_A[reg_stage][m_tile (0..1)][4 regs]
    // frag_B[reg_stage][n_pair (0..1)][4 regs]
    uint32_t frag_A[2][2][4];
    uint32_t frag_B[2][2][4];

    // FP32 master accumulators across K-groups (2 m16 tiles x 4 n8 tiles x 4 regs)
    float acc_f32[2][4][4];
    #pragma unroll
    for (int m = 0; m < 2; ++m) {
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            acc_f32[m][n][0] = 0.0f;
            acc_f32[m][n][1] = 0.0f;
            acc_f32[m][n][2] = 0.0f;
            acc_f32[m][n][3] = 0.0f;
        }
    }

    // ------------------------------------------------------------------------
    // Prologue: Issue cp.async for k_tile = 0 into smem_stage = 0
    // ------------------------------------------------------------------------
    {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            int row_a = c * 16 + load_row_in_chunk;
            int g_row_a = cta_m_base + row_a;
            uint32_t smem_a = cvta_to_shared_u32(
                &smem_buf[0][0][smem_offset_swizzle_3_4_3(row_a, load_col_16B)]
            );
            const uint8_t* gmem_a = A_packed + static_cast<int64_t>(g_row_a) * k_bytes + (load_col_16B << 4);
            cp_async_cg_16B_pred(smem_a, gmem_a, g_row_a < M);

            int row_b = c * 16 + load_row_in_chunk;
            int g_row_b = cta_n_base + row_b;
            uint32_t smem_b = cvta_to_shared_u32(
                &smem_buf[0][1][smem_offset_swizzle_3_4_3(row_b, load_col_16B)]
            );
            const uint8_t* gmem_b = B_packed + static_cast<int64_t>(g_row_b) * k_bytes + (load_col_16B << 4);
            cp_async_cg_16B_pred(smem_b, gmem_b, g_row_b < N);
        }
        cp_async_commit();
    }

    // Wait for k_tile = 0 to land in smem_stage = 0
    cp_async_wait_group<0>();
    __syncthreads();

    // Load k_step = 0 of k_tile = 0 into reg_stage = 0
    #pragma unroll
    for (int m = 0; m < 2; ++m) {
        int row = warp_m_idx * 32 + m * 16 + a_lane_row_offset;
        int col = a_lane_col_offset;
        uint32_t addr = cvta_to_shared_u32(&smem_buf[0][0][smem_offset_swizzle_3_4_3(row, col)]);
        ldmatrix_m8n8_x4_b16(frag_A[0][m][0], frag_A[0][m][1], frag_A[0][m][2], frag_A[0][m][3], addr);
    }
    #pragma unroll
    for (int np = 0; np < 2; ++np) {
        int row = warp_n_idx * 32 + np * 16 + b_lane_row_offset;
        int col = b_lane_col_offset;
        uint32_t addr = cvta_to_shared_u32(&smem_buf[0][1][smem_offset_swizzle_3_4_3(row, col)]);
        ldmatrix_m8n8_x4_b16(frag_B[0][np][0], frag_B[0][np][1], frag_B[0][np][2], frag_B[0][np][3], addr);
    }

    // ------------------------------------------------------------------------
    // Mainloop: Double-Buffered Shared Memory + Double-Buffered Register Pipeline
    // ------------------------------------------------------------------------
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        int smem_read_stage  = k_tile & 1;
        int smem_write_stage = smem_read_stage ^ 1;

        // Prefetch next K-tile (k_tile + 1) into smem_write_stage
        if (k_tile + 1 < num_k_tiles) {
            int next_k_byte_offset = (k_tile + 1) << 7; // (k_tile + 1) * 128B
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                int row_a = c * 16 + load_row_in_chunk;
                int g_row_a = cta_m_base + row_a;
                uint32_t smem_a = cvta_to_shared_u32(
                    &smem_buf[smem_write_stage][0][smem_offset_swizzle_3_4_3(row_a, load_col_16B)]
                );
                const uint8_t* gmem_a = A_packed
                    + static_cast<int64_t>(g_row_a) * k_bytes
                    + next_k_byte_offset + (load_col_16B << 4);
                cp_async_cg_16B_pred(smem_a, gmem_a, g_row_a < M);

                int row_b = c * 16 + load_row_in_chunk;
                int g_row_b = cta_n_base + row_b;
                uint32_t smem_b = cvta_to_shared_u32(
                    &smem_buf[smem_write_stage][1][smem_offset_swizzle_3_4_3(row_b, load_col_16B)]
                );
                const uint8_t* gmem_b = B_packed
                    + static_cast<int64_t>(g_row_b) * k_bytes
                    + next_k_byte_offset + (load_col_16B << 4);
                cp_async_cg_16B_pred(smem_b, gmem_b, g_row_b < N);
            }
            cp_async_commit();
        }

        // INT32 accumulators for the current 256-element H_256 group
        int32_t acc_s32[2][4][4];
        #pragma unroll
        for (int m = 0; m < 2; ++m) {
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                acc_s32[m][n][0] = 0;
                acc_s32[m][n][1] = 0;
                acc_s32[m][n][2] = 0;
                acc_s32[m][n][3] = 0;
            }
        }

        // 4 MMA k-steps (4 * 64 = 256 INT4 elements) with register ping-pong
        #pragma unroll
        for (int k_step = 0; k_step < 4; ++k_step) {
            int reg_read  = k_step & 1;
            int reg_write = reg_read ^ 1;

            // At k_step == 3, synchronize next shared memory stage so we can
            // preload k_step = 0 of (k_tile + 1) before issuing the final MMA!
            if (k_step == 3) {
                if (k_tile + 1 < num_k_tiles) {
                    cp_async_wait_group<0>();
                }
                __syncthreads();
            }

            // Preload next register buffer
            if (k_step < 3) {
                int next_k_col_base = (k_step + 1) << 1;
                #pragma unroll
                for (int m = 0; m < 2; ++m) {
                    int row = warp_m_idx * 32 + m * 16 + a_lane_row_offset;
                    int col = next_k_col_base + a_lane_col_offset;
                    uint32_t addr = cvta_to_shared_u32(
                        &smem_buf[smem_read_stage][0][smem_offset_swizzle_3_4_3(row, col)]
                    );
                    ldmatrix_m8n8_x4_b16(
                        frag_A[reg_write][m][0], frag_A[reg_write][m][1],
                        frag_A[reg_write][m][2], frag_A[reg_write][m][3], addr
                    );
                }
                #pragma unroll
                for (int np = 0; np < 2; ++np) {
                    int row = warp_n_idx * 32 + np * 16 + b_lane_row_offset;
                    int col = next_k_col_base + b_lane_col_offset;
                    uint32_t addr = cvta_to_shared_u32(
                        &smem_buf[smem_read_stage][1][smem_offset_swizzle_3_4_3(row, col)]
                    );
                    ldmatrix_m8n8_x4_b16(
                        frag_B[reg_write][np][0], frag_B[reg_write][np][1],
                        frag_B[reg_write][np][2], frag_B[reg_write][np][3], addr
                    );
                }
            } else if (k_tile + 1 < num_k_tiles) {
                #pragma unroll
                for (int m = 0; m < 2; ++m) {
                    int row = warp_m_idx * 32 + m * 16 + a_lane_row_offset;
                    int col = a_lane_col_offset;
                    uint32_t addr = cvta_to_shared_u32(
                        &smem_buf[smem_write_stage][0][smem_offset_swizzle_3_4_3(row, col)]
                    );
                    ldmatrix_m8n8_x4_b16(
                        frag_A[0][m][0], frag_A[0][m][1],
                        frag_A[0][m][2], frag_A[0][m][3], addr
                    );
                }
                #pragma unroll
                for (int np = 0; np < 2; ++np) {
                    int row = warp_n_idx * 32 + np * 16 + b_lane_row_offset;
                    int col = b_lane_col_offset;
                    uint32_t addr = cvta_to_shared_u32(
                        &smem_buf[smem_write_stage][1][smem_offset_swizzle_3_4_3(row, col)]
                    );
                    ldmatrix_m8n8_x4_b16(
                        frag_B[0][np][0], frag_B[0][np][1],
                        frag_B[0][np][2], frag_B[0][np][3], addr
                    );
                }
            }

            // Issue m16n8k64 s4 Tensor Core MMAs on reg_read
            #pragma unroll
            for (int m = 0; m < 2; ++m) {
                #pragma unroll
                for (int np = 0; np < 2; ++np) {
                    mma_m16n8k64_s4(
                        acc_s32[m][np * 2 + 0][0], acc_s32[m][np * 2 + 0][1],
                        acc_s32[m][np * 2 + 0][2], acc_s32[m][np * 2 + 0][3],
                        frag_A[reg_read][m][0], frag_A[reg_read][m][1],
                        frag_A[reg_read][m][2], frag_A[reg_read][m][3],
                        frag_B[reg_read][np][0], frag_B[reg_read][np][1]
                    );
                    mma_m16n8k64_s4(
                        acc_s32[m][np * 2 + 1][0], acc_s32[m][np * 2 + 1][1],
                        acc_s32[m][np * 2 + 1][2], acc_s32[m][np * 2 + 1][3],
                        frag_A[reg_read][m][0], frag_A[reg_read][m][1],
                        frag_A[reg_read][m][2], frag_A[reg_read][m][3],
                        frag_B[reg_read][np][2], frag_B[reg_read][np][3]
                    );
                }
            }
        }

        // --------------------------------------------------------------------
        // Per-Group (G = 256) Dequantization & Accumulation into acc_f32
        // --------------------------------------------------------------------
        const int group_id     = lane_id >> 2;
        const int tid_in_group = lane_id & 3;

        float sb_cache[4][2];
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            int col0 = cta_n_base + warp_n_idx * 32 + n * 8 + tid_in_group * 2;
            int col1 = col0 + 1;
            sb_cache[n][0] = (col0 < N) ? scale_B[col0 * num_k_tiles + k_tile] : 0.0f;
            sb_cache[n][1] = (col1 < N) ? scale_B[col1 * num_k_tiles + k_tile] : 0.0f;
        }

        #pragma unroll
        for (int m = 0; m < 2; ++m) {
            int row0 = cta_m_base + warp_m_idx * 32 + m * 16 + group_id;
            int row1 = row0 + 8;
            float sa0 = (row0 < M) ? scale_A[row0 * num_k_tiles + k_tile] : 0.0f;
            float sa1 = (row1 < M) ? scale_A[row1 * num_k_tiles + k_tile] : 0.0f;

            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                float s00 = sa0 * sb_cache[n][0];
                float s01 = sa0 * sb_cache[n][1];
                float s10 = sa1 * sb_cache[n][0];
                float s11 = sa1 * sb_cache[n][1];

                acc_f32[m][n][0] = __fmaf_rn(static_cast<float>(acc_s32[m][n][0]), s00, acc_f32[m][n][0]);
                acc_f32[m][n][1] = __fmaf_rn(static_cast<float>(acc_s32[m][n][1]), s01, acc_f32[m][n][1]);
                acc_f32[m][n][2] = __fmaf_rn(static_cast<float>(acc_s32[m][n][2]), s10, acc_f32[m][n][2]);
                acc_f32[m][n][3] = __fmaf_rn(static_cast<float>(acc_s32[m][n][3]), s11, acc_f32[m][n][3]);
            }
        }
    }

    // ------------------------------------------------------------------------
    // Epilogue: Convert acc_f32 to FP16 and Write Back to Global Memory C
    // ------------------------------------------------------------------------
    const int group_id     = lane_id >> 2;
    const int tid_in_group = lane_id & 3;

    #pragma unroll
    for (int m = 0; m < 2; ++m) {
        int row0 = cta_m_base + warp_m_idx * 32 + m * 16 + group_id;
        int row1 = row0 + 8;

        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            int col0 = cta_n_base + warp_n_idx * 32 + n * 8 + tid_in_group * 2;

            if (row0 < M && (col0 + 1) < N) {
                *reinterpret_cast<uint32_t*>(&C_f16[static_cast<int64_t>(row0) * N + col0]) =
                    float2_to_half2(acc_f32[m][n][0], acc_f32[m][n][1]);
            } else if (row0 < M && col0 < N) {
                C_f16[static_cast<int64_t>(row0) * N + col0] = float_to_half(acc_f32[m][n][0]);
            }

            if (row1 < M && (col0 + 1) < N) {
                *reinterpret_cast<uint32_t*>(&C_f16[static_cast<int64_t>(row1) * N + col0]) =
                    float2_to_half2(acc_f32[m][n][2], acc_f32[m][n][3]);
            } else if (row1 < M && col0 < N) {
                C_f16[static_cast<int64_t>(row1) * N + col0] = float_to_half(acc_f32[m][n][2]);
            }
        }
    }
}

// ============================================================================
// Section 6: Host Launch Functions (Guarded for NVCC / C-FFI; Ignored by NVRTC)
// ============================================================================
#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>

extern "C" cudaError_t launch_w4a4_h256_dynamic_quant(
    const void* input_f16,
    void* output_packed_s4,
    float* output_scales,
    int num_groups,
    cudaStream_t stream
) {
    const int threads_per_block = 256; // 8 warps = 8 groups of 256 per CTA
    const int warps_per_block   = threads_per_block >> 5;
    const int blocks            = (num_groups + warps_per_block - 1) / warps_per_block;

    w4a4_h256_dynamic_quant_kernel<<<blocks, threads_per_block, 0, stream>>>(
        reinterpret_cast<const uint4*>(input_f16),
        reinterpret_cast<uint32_t*>(output_packed_s4),
        output_scales,
        num_groups
    );
    return cudaGetLastError();
}

extern "C" cudaError_t launch_w4a4_ampere_gemm_swizzle_pipeline(
    const void* A_packed,
    const void* B_packed,
    const float* scale_A,
    const float* scale_B,
    void* C_f16,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    if ((K & 255) != 0) {
        return cudaErrorInvalidValue;
    }

    dim3 block(128, 1, 1);
    dim3 grid((N + 63) / 64, (M + 63) / 64, 1);

    w4a4_ampere_gemm_swizzle_pipeline_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(A_packed),
        reinterpret_cast<const uint8_t*>(B_packed),
        scale_A,
        scale_B,
        reinterpret_cast<uint16_t*>(C_f16),
        M,
        N,
        K
    );
    return cudaGetLastError();
}
#endif
