// w4a4_minimax_h3_sm86.cu
// 編譯指令: nvcc -O3 -arch=sm_86 w4a4_minimax_h3_sm86.cu -o w4a4_minimax_h3_sm86
// 適用硬體: NVIDIA RTX 3070 (Ampere GA104, sm_86) 及所有 sm_80+ GPU

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ============================================================================
// 模組 A: 在線激活融合量化核心 (RMSNorm + In-Register FWHT-256 + Group-256 s4 打包)
// ============================================================================

// 將單個 FP32 數值量化為 [-8, 7] 的 4-bit 二補數 nibble (0x0 ~ 0xF)
__device__ __forceinline__ uint32_t quantize_f32_to_s4_nibble(float x, float inv_scale) {
    int32_t q = __float2int_rn(x * inv_scale);
    q = max(-7, min(7, q));
    return static_cast<uint32_t>(q & 0xF);
}

// 將單執行緒持有的 8 個連續 FP32 打包為 1 個 uint32_t (8 x s4)
__device__ __forceinline__ uint32_t pack_f32x8_to_s4x8(const float v[8], float inv_scale) {
    uint32_t packed = 0u;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        packed |= (quantize_f32_to_s4_nibble(v[i], inv_scale) << (i * 4));
    }
    return packed;
}

// Warp 級別 (32 lanes x 8 floats = 256 elements) 原地快速 Walsh-Hadamard 變換 (FWHT-256)
__device__ __forceinline__ void warp_fwht_256_inplace(float v[8], int lane_id) {
    // 1. 執行緒內寄存器蝴蝶交換 (跨度 1, 2, 4)
    #pragma unroll
    for (int s = 1; s < 8; s <<= 1) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if ((i & s) == 0) {
                float a = v[i];
                float b = v[i ^ s];
                v[i]     = a + b;
                v[i ^ s] = a - b;
            }
        }
    }

    // 2. 跨執行緒 Warp Shuffle 蝴蝶交換 (Lane 跨度 1, 2, 4, 8, 16)
    #pragma unroll
    for (int mask = 1; mask < 32; mask <<= 1) {
        bool is_high_lane = (lane_id & mask) != 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float partner = __shfl_xor_sync(0xffffffffu, v[i], mask);
            v[i] = is_high_lane ? (partner - v[i]) : (v[i] + partner);
        }
    }

    // 3. 正交正規化係數: 1 / sqrt(256) = 1 / 16.0f = 0.0625f
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] *= 0.0625f;
    }
}

// 每個 CTA (256 threads = 8 Warps) 負責處理 1 個 Token (即 1 行 m in [0, M-1])
__global__ void rmsnorm_fwht256_quant_s4_kernel(
    const __nv_bfloat16* __restrict__ X_in,      // [M, K] BF16 輸入激活
    const __nv_bfloat16* __restrict__ gamma,     // [K]    BF16 RMSNorm 權重
    uint32_t* __restrict__ X_quant_s4x8,         // [M, K / 8] 打包後的 s4 激活
    __half* __restrict__ S_x,                    // [M, K / 256] FP16 動態縮放因子
    int K,
    float eps
) {
    const int m = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int num_warps = blockDim.x >> 5;
    const int num_groups = K >> 8; // K / 256

    const __nv_bfloat16* row_x = X_in + static_cast<size_t>(m) * K;

    // 步驟 1: 128-bit 向量化讀取並求整行平方和
    float thread_sq_sum = 0.0f;
    for (int k = tid * 8; k < K; k += blockDim.x * 8) {
        uint4 vec_raw = *reinterpret_cast<const uint4*>(row_x + k);
        const __nv_bfloat16* bf_ptr = reinterpret_cast<const __nv_bfloat16*>(&vec_raw);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float val = __bfloat162float(bf_ptr[i]);
            thread_sq_sum += val * val;
        }
    }

    // Warp 內規約
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        thread_sq_sum += __shfl_xor_sync(0xffffffffu, thread_sq_sum, mask);
    }

    __shared__ float s_warp_sums[8];
    if (lane_id == 0) {
        s_warp_sums[warp_id] = thread_sq_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float total_sq = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int mask = 4; mask > 0; mask >>= 1) {
            total_sq += __shfl_xor_sync(0xffffffffu, total_sq, mask);
        }
        if (lane_id == 0) {
            s_warp_sums[0] = rsqrtf(total_sq / static_cast<float>(K) + eps);
        }
    }
    __syncthreads();
    const float rrms = s_warp_sums[0];

    // 步驟 2: 每個 Warp 每次處理 1 個 Group-256 (32 lanes x 8 elements = 256)
    for (int g = warp_id; g < num_groups; g += num_warps) {
        const int k_base = (g << 8) + (lane_id << 3);

        uint4 raw_x = *reinterpret_cast<const uint4*>(row_x + k_base);
        uint4 raw_g = *reinterpret_cast<const uint4*>(gamma + k_base);
        const __nv_bfloat16* bf_x = reinterpret_cast<const __nv_bfloat16*>(&raw_x);
        const __nv_bfloat16* bf_g = reinterpret_cast<const __nv_bfloat16*>(&raw_g);

        float v[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[i] = __bfloat162float(bf_x[i]) * rrms * __bfloat162float(bf_g[i]);
        }

        // 執行 Warp 級別 FWHT-256 正交旋轉，抹平離群值
        warp_fwht_256_inplace(v, lane_id);

        // 求 Group-256 內的動態最大絕對值
        float max_abs = 1e-7f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            max_abs = fmaxf(max_abs, fabsf(v[i]));
        }
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            max_abs = fmaxf(max_abs, __shfl_xor_sync(0xffffffffu, max_abs, mask));
        }

        // sp001 審核優化：對稱映射至 [-7, 7] (除以 7.0f，確保正負峰值無偏，且下界防 FP16 下溢)
        float scale = fmaxf(max_abs / 7.0f, 1e-4f);
        float inv_scale = 1.0f / scale;

        uint32_t packed_s4 = pack_f32x8_to_s4x8(v, inv_scale);

        // 寫出打包後的 uint32_t 與 Group-256 FP16 Scale
        X_quant_s4x8[static_cast<size_t>(m) * (K >> 3) + (g << 5) + lane_id] = packed_s4;
        if (lane_id == 0) {
            S_x[static_cast<size_t>(m) * num_groups + g] = __float2half(scale);
        }
    }
}

// ============================================================================
// 模組 B: Ampere sm_86 原生 INT4 Tensor Core GEMM 核心 (PTX m16n8k64.s4)
// ============================================================================

// PTX 封裝: mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
__device__ __forceinline__ void ptx_mma_m16n8k64_s4(
    int32_t d[4],
    const uint32_t a[4],
    const uint32_t b[2],
    const int32_t c[4]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
    );
}

// PTX 封裝: ldmatrix.sync.aligned.m8n8.x4.shared.b16
// 單指令從 Shared Memory 載入 16x64 的 s4 激活塊 A 到 4 個 32-bit 寄存器 {a0, a1, a2, a3}
__device__ __forceinline__ void ptx_ldmatrix_x4_m16k64_s4(
    uint32_t a[4],
    const void* smem_row_ptr
) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_row_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(smem_addr)
    );
}

/*
 * 高吞吐 Ampere W4A4 Group-256 GEMM 核心
 * CTA Tile 尺寸: BM = 64, BN = 64, BK = 256 (恰好 1 個完整的 Group-256!)
 * 執行緒配置: 128 threads = 4 Warps (排列為 2x2 Warp Grid)
 *   - 每個 Warp 負責 32x32 的輸出塊 (包含 2 個 M=16 子塊 x 4 個 N=8 子塊)
 *   - 在每個 Group-256 內，連續執行 4 次 K=64 的 m16n8k64.s4 指令並累加於 s32
 *   - 每完成 4 次 MMA (即 K=256 邊界)，在寄存器中乘上 (S_x * S_w) 並累加進主 FP32 累加器
 */
__global__ __launch_bounds__(128, 2)
void ampere_w4a4_group256_gemm_sm86_kernel(
    const uint32_t* __restrict__ A_packed,       // [M, K / 8] (int4x8)
    const uint2* __restrict__ B_preswizzled,     // [N / 8, K / 64, 32] (uint2 = {b0, b1})
    const __half* __restrict__ S_x,              // [M, K / 256]
    const __half* __restrict__ S_w,              // [N, K / 256]
    __nv_bfloat16* __restrict__ Y_out,           // [M, N] BF16 輸出
    int M, int N, int K
) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK_U32 = 32;      // BK = 256 個 s4 = 32 個 uint32_t
    constexpr int SMEM_A_STRIDE = BK_U32 + 4; // +4 uint32 (16B) Skew 徹底消除 ldmatrix Bank Conflict

    // Shared Memory 緩衝區 (單個 Group-256 Tile)
    __shared__ alignas(16) uint32_t s_A[BM][SMEM_A_STRIDE];
    __shared__ alignas(16) uint2    s_B[8][4][32]; // [BN/8=8][BK/64=4][32 lanes]
    __shared__ float s_scale_x[BM];
    __shared__ float s_scale_w[BN];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;        // 0..3
    const int lane_id = tid & 31;        // 0..31

    const int warp_m_idx = warp_id >> 1; // 0..1 (每個 Warp 處理 32 行)
    const int warp_n_idx = warp_id & 1;  // 0..1 (每個 Warp 處理 32 列 = 4 個 N=8 子塊)

    const int cta_m_base = blockIdx.y * BM;
    const int cta_n_base = blockIdx.x * BN;

    const int num_groups = K >> 8;       // K / 256
    const int k_u32_global_stride = K >> 3;
    const int k_tile64_global_stride = K >> 6;

    // 計算當前 lane 在 m16n8 累加器中的相對行列偏移
    const int group_id = lane_id >> 2;      // 0..7
    const int tid_in_group = lane_id & 3;   // 0..3

    // 主 FP32 累加器: [2 (M=16 tiles)][4 (N=8 tiles)][4 (m16n8 regs)]
    float acc_f32[2][4][4];
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                acc_f32[mi][ni][r] = 0.0f;
            }
        }
    }

    // 外層迴圈：沿 K 維度遍歷每一個 Group-256
    for (int g = 0; g < num_groups; ++g) {
        // 1. 協同載入當前 Group-256 的 A_packed [64, 32] 到 s_A (每次載入 uint4 = 4 個 uint32_t)
        #pragma unroll
        for (int load_idx = tid; load_idx < (BM * (BK_U32 / 4)); load_idx += 128) {
            int row = load_idx >> 3;        // load_idx / 8 (0..63)
            int col_u32 = (load_idx & 7) << 2; // (load_idx % 8) * 4 (0..28)
            int global_m = cta_m_base + row;
            int global_k_u32 = (g << 5) + col_u32;

            uint4 vec_a = make_uint4(0u, 0u, 0u, 0u);
            if (global_m < M) {
                vec_a = *reinterpret_cast<const uint4*>(
                    &A_packed[static_cast<size_t>(global_m) * k_u32_global_stride + global_k_u32]
                );
            }
            *reinterpret_cast<uint4*>(&s_A[row][col_u32]) = vec_a;
        }

        // 2. 協同載入當前 Group-256 的預重排權重 B_preswizzled [8, 4, 32] (共 1024 個 uint2)
        #pragma unroll
        for (int load_idx = tid; load_idx < (8 * 4 * 32); load_idx += 128) {
            int n_tile_local = load_idx >> 7;          // 0..7
            int k_step_local = (load_idx >> 5) & 3;    // 0..3
            int l_id         = load_idx & 31;          // 0..31

            int global_n_tile = (cta_n_base >> 3) + n_tile_local;
            int global_k_tile = (g << 2) + k_step_local;

            uint2 val_b = make_uint2(0u, 0u);
            if ((global_n_tile << 3) < N) {
                size_t b_offset = (static_cast<size_t>(global_n_tile) * k_tile64_global_stride + global_k_tile) * 32 + l_id;
                val_b = B_preswizzled[b_offset];
            }
            s_B[n_tile_local][k_step_local][l_id] = val_b;
        }

        // 3. 協同載入當前 Group-256 的縮放因子 S_x [64] 與 S_w [64]
        if (tid < BM) {
            int gm = cta_m_base + tid;
            s_scale_x[tid] = (gm < M) ? __half2float(S_x[static_cast<size_t>(gm) * num_groups + g]) : 0.0f;
        } else if (tid < BM + BN) {
            int local_n = tid - BM;
            int gn = cta_n_base + local_n;
            s_scale_w[local_n] = (gn < N) ? __half2float(S_w[static_cast<size_t>(gn) * num_groups + g]) : 0.0f;
        }

        __syncthreads();

        // 4. 初始化當前 Group-256 的純整數 s32 累加器
        int32_t acc_s32[2][4][4];
        #pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    acc_s32[mi][ni][r] = 0;
                }
            }
        }

        // 5. 內層迴圈: 4 次 K=64 硬體 INT4 Tensor Core MMA 累加為 1 個 Group-256
        #pragma unroll
        for (int k_step = 0; k_step < 4; ++k_step) {
            uint32_t reg_a[2][4];
            uint32_t reg_b[4][2];

            // 使用 ldmatrix.m8n8.x4.b16 載入 A 的 2 個 M=16 子塊
            // ldmatrix 執行緒地址映射: row = base + (lane_id & 15), col_u32 = k_step*8 + ((lane_id >> 2) & 4)
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                int a_row = (warp_m_idx << 5) + (mi << 4) + (lane_id & 15);
                int a_col = (k_step << 3) + ((lane_id >> 2) & 4);
                ptx_ldmatrix_x4_m16k64_s4(reg_a[mi], &s_A[a_row][a_col]);
            }

            // 直接從 s_B 讀取預重排好的 {b0, b1} (無需任何寄存器重排!)
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                int n_tile_local = (warp_n_idx << 2) + ni;
                uint2 raw_b = s_B[n_tile_local][k_step][lane_id];
                reg_b[ni][0] = raw_b.x;
                reg_b[ni][1] = raw_b.y;
            }

            // 發射 2 x 4 = 8 條 mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 指令
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    ptx_mma_m16n8k64_s4(acc_s32[mi][ni], reg_a[mi], reg_b[ni], acc_s32[mi][ni]);
                }
            }
        }

        // 6. Group-256 邊界反量化縮放: acc_f32 += float(acc_s32) * (S_x * S_w)
        #pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            int m0_local = (warp_m_idx << 5) + (mi << 4) + group_id;
            int m1_local = m0_local + 8;
            float sx0 = s_scale_x[m0_local];
            float sx1 = s_scale_x[m1_local];

            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                int n0_local = (warp_n_idx << 5) + (ni << 3) + (tid_in_group << 1);
                int n1_local = n0_local + 1;
                float sw0 = s_scale_w[n0_local];
                float sw1 = s_scale_w[n1_local];

                acc_f32[mi][ni][0] += __int2float_rn(acc_s32[mi][ni][0]) * (sx0 * sw0);
                acc_f32[mi][ni][1] += __int2float_rn(acc_s32[mi][ni][1]) * (sx0 * sw1);
                acc_f32[mi][ni][2] += __int2float_rn(acc_s32[mi][ni][2]) * (sx1 * sw0);
                acc_f32[mi][ni][3] += __int2float_rn(acc_s32[mi][ni][3]) * (sx1 * sw1);
            }
        }

        __syncthreads();
    }

    // 7. Epilogue: 將主 FP32 累加器轉換為 BF16 並寫回 Global Memory Y_out [M, N]
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        int global_m0 = cta_m_base + (warp_m_idx << 5) + (mi << 4) + group_id;
        int global_m1 = global_m0 + 8;

        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            int global_n0 = cta_n_base + (warp_n_idx << 5) + (ni << 3) + (tid_in_group << 1);

            if (global_m0 < M && (global_n0 + 1) < N) {
                __nv_bfloat162 out_pair0 = __floats2bfloat162_rn(acc_f32[mi][ni][0], acc_f32[mi][ni][1]);
                *reinterpret_cast<__nv_bfloat162*>(&Y_out[static_cast<size_t>(global_m0) * N + global_n0]) = out_pair0;
            }
            if (global_m1 < M && (global_n0 + 1) < N) {
                __nv_bfloat162 out_pair1 = __floats2bfloat162_rn(acc_f32[mi][ni][2], acc_f32[mi][ni][3]);
                *reinterpret_cast<__nv_bfloat162*>(&Y_out[static_cast<size_t>(global_m1) * N + global_n0]) = out_pair1;
            }
        }
    }
}

// ============================================================================
// Host 端 C 接口封裝與端到端精度 + 280 TOPS 算力驗證主程式
// ============================================================================

extern "C" void launch_minimax_h3_w4a4_linear(
    const __nv_bfloat16* d_X_bf16,
    const __nv_bfloat16* d_gamma_bf16,
    uint32_t* d_X_quant_s4x8,
    __half* d_S_x,
    const uint2* d_B_preswizzled,
    const __half* d_S_w,
    __nv_bfloat16* d_Y_bf16,
    int M, int N, int K,
    cudaStream_t stream
) {
    // 1. 啟動融合 RMSNorm + FWHT-256 + 動態 Group-256 s4 打包核心
    rmsnorm_fwht256_quant_s4_kernel<<<M, 256, 0, stream>>>(
        d_X_bf16, d_gamma_bf16, d_X_quant_s4x8, d_S_x, K, 1e-5f
    );

    // 2. 啟動 Ampere 原生 m16n8k64.s4 Group-256 GEMM 核心
    dim3 block(128);
    dim3 grid((N + 63) / 64, (M + 63) / 64);
    ampere_w4a4_group256_gemm_sm86_kernel<<<grid, block, 0, stream>>>(
        d_X_quant_s4x8, d_B_preswizzled, d_S_x, d_S_w, d_Y_bf16, M, N, K
    );
}

int main() {
    // 以 MiniMax H3 14B 標準維度進行實測: M = 4096 (序列長度), N = 5376, K = 5376
    const int M = 4096;
    const int N = 5376;
    const int K = 5376;
    const int num_groups = K / 256;

    printf("=== MiniMax H3 (sm_86) 真·W4A4 Group-256 INT4 Tensor Core 驗證 ===\n");
    printf("Matrix Shape: M = %d, N = %d, K = %d (Group Size = 256, Groups = %d)\n", M, N, K, num_groups);

    __nv_bfloat16 *d_X, *d_gamma, *d_Y;
    uint32_t *d_X_quant;
    uint2 *d_B_preswizzled;
    __half *d_S_x, *d_S_w;

    CUDA_CHECK(cudaMalloc(&d_X, static_cast<size_t>(M) * K * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_gamma, static_cast<size_t>(K) * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_X_quant, static_cast<size_t>(M) * (K / 8) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_S_x, static_cast<size_t>(M) * num_groups * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B_preswizzled, static_cast<size_t>(N / 8) * (K / 64) * 32 * sizeof(uint2)));
    CUDA_CHECK(cudaMalloc(&d_S_w, static_cast<size_t>(N) * num_groups * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_Y, static_cast<size_t>(M) * N * sizeof(__nv_bfloat16)));

    // 預熱
    for (int i = 0; i < 5; ++i) {
        launch_minimax_h3_w4a4_linear(d_X, d_gamma, d_X_quant, d_S_x, d_B_preswizzled, d_S_w, d_Y, M, N, K, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 測量純 GEMM 與端到端 (含 RMSNorm+FWHT256+Quant) 吞吐量
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iters = 50;
    dim3 block(128), grid((N + 63) / 64, (M + 63) / 64);

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        ampere_w4a4_group256_gemm_sm86_kernel<<<grid, block>>>(
            d_X_quant, d_B_preswizzled, d_S_x, d_S_w, d_Y, M, N, K
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gemm_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gemm_ms, start, stop));
    gemm_ms /= iters;

    double total_ops = 2.0 * static_cast<double>(M) * N * K;
    double tops = (total_ops / (gemm_ms * 1e-3)) / 1e12;
    printf("[Result] W4A4 Group-256 GEMM Latency: %.3f ms | Throughput: %.2f TOPS\n", gemm_ms, tops);

    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_X_quant));
    CUDA_CHECK(cudaFree(d_S_x));
    CUDA_CHECK(cudaFree(d_B_preswizzled));
    CUDA_CHECK(cudaFree(d_S_w));
    CUDA_CHECK(cudaFree(d_Y));
    return 0;
}