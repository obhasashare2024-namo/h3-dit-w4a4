#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define WARMUP 100
#define ITERS 5000
#define UNROLL 16

__global__ void bench_mma_m16n8k32_s8(int iters, float *out_dummy) {
    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    uint32_t a0 = 0x01010101, a1 = 0x01010101, a2 = 0x01010101, a3 = 0x01010101;
    uint32_t b0 = 0x01010101, b1 = 0x01010101;

    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
            );
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out_dummy = (float)(d0 + d1 + d2 + d3);
    }
}

__global__ void bench_mma_m16n8k64_s4(int iters, float *out_dummy) {
    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    uint32_t a0 = 0x11111111, a1 = 0x11111111, a2 = 0x11111111, a3 = 0x11111111;
    uint32_t b0 = 0x11111111, b1 = 0x11111111;

    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
            );
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out_dummy = (float)(d0 + d1 + d2 + d3);
    }
}

__global__ void bench_mma_m16n8k16_f16(int iters, float *out_dummy) {
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    uint32_t a0 = 0x3c003c00, a1 = 0x3c003c00, a2 = 0x3c003c00, a3 = 0x3c003c00;
    uint32_t b0 = 0x3c003c00, b1 = 0x3c003c00;

    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
            );
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out_dummy = d0 + d1 + d2 + d3;
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    int num_sms = prop.multiProcessorCount;
    int threads_per_block = 128;
    int blocks = num_sms * 4;

    float *d_dummy;
    cudaMalloc(&d_dummy, sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double total_mma = (double)blocks * (threads_per_block / 32) * ITERS * UNROLL;

    // 1. FP16
    bench_mma_m16n8k16_f16<<<blocks, threads_per_block>>>(WARMUP, d_dummy);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    bench_mma_m16n8k16_f16<<<blocks, threads_per_block>>>(ITERS, d_dummy);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_f16 = 0;
    cudaEventElapsedTime(&ms_f16, start, stop);
    double ops_f16 = total_mma * 2.0 * 16 * 8 * 16;
    double tflops_f16 = (ops_f16 / (ms_f16 * 1e-3)) / 1e12;

    // 2. INT8
    bench_mma_m16n8k32_s8<<<blocks, threads_per_block>>>(WARMUP, d_dummy);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    bench_mma_m16n8k32_s8<<<blocks, threads_per_block>>>(ITERS, d_dummy);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_s8 = 0;
    cudaEventElapsedTime(&ms_s8, start, stop);
    double ops_s8 = total_mma * 2.0 * 16 * 8 * 32;
    double tops_s8 = (ops_s8 / (ms_s8 * 1e-3)) / 1e12;

    // 3. INT4
    bench_mma_m16n8k64_s4<<<blocks, threads_per_block>>>(WARMUP, d_dummy);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    bench_mma_m16n8k64_s4<<<blocks, threads_per_block>>>(ITERS, d_dummy);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_s4 = 0;
    cudaEventElapsedTime(&ms_s4, start, stop);
    double ops_s4 = total_mma * 2.0 * 16 * 8 * 64;
    double tops_s4 = (ops_s4 / (ms_s4 * 1e-3)) / 1e12;

    printf("\n=== Raw Tensor Core MMA Microbenchmark (sm_86) ===\n");
    printf("1. m16n8k16.f16 (FP16): %8.3f ms | %6.2f TFLOPS\n", ms_f16, tflops_f16);
    printf("2. m16n8k32.s8  (INT8): %8.3f ms | %6.2f TOPS  (Ratio vs FP16: %.2fx)\n", ms_s8, tops_s8, tops_s8 / tflops_f16);
    printf("3. m16n8k64.s4  (INT4): %8.3f ms | %6.2f TOPS  (Ratio vs INT8: %.2fx)\n", ms_s4, tops_s4, tops_s4 / tops_s8);
    printf("Instruction Latency Ratio (s4/s8): %.3fx (lower = faster per MMA instruction)\n", ms_s4 / ms_s8);

    cudaFree(d_dummy);
    return 0;
}
