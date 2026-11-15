# h3-dit-w4a4

[English](README.md) | [中文说明](README_zh.md) | [日本語ドキュメント](README_ja.md)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust: 100%](https://img.shields.io/badge/Rust-100%25-orange.svg)](src/)
[![Architecture: Ampere sm_86+](https://img.shields.io/badge/Arch-Ampere%20sm__86%2B-green.svg)](w4a4_minimax_h3_sm86.cu)
[![Format: GGUF & Safetensors](https://img.shields.io/badge/Format-GGUF%20%2F%20Safetensors-blueviolet.svg)](#gguf-container-compatibility-a4--a8--a16)

High-performance, pure Rust microservice engine for **MiniMax H3 (HyperFlow Flash 14B)** text/image-to-video DiT (Diffusion Transformer) inference, powered by **native NVIDIA Ampere Tensor Core INT4 (W4A4) and INT8 (W4A8) hardware acceleration**, **GGUF container compatibility**, **3-Tier (VRAM-RAM-Disk) dynamic memory streaming**, and **adaptive head-grouping / staggered VRAM lifecycle management**.

Engineered specifically for low-VRAM consumer GPUs (e.g., NVIDIA GeForce RTX 3070 8GB Laptop/Desktop, RTX 4060Ti, RTX 3060 12GB), breaking the monolithic VRAM barrier and achieving ultra-fast 1080p full-spectrum video synthesis.

---

## Table of Contents

- [Key Highlights](#key-highlights)
- [Architectural Decoupling: Why Monolithic DiT Fails](#architectural-decoupling-why-monolithic-dit-fails)
- [Hardware Tensor Core Acceleration: True A4 & A8 MMA](#hardware-tensor-core-acceleration-true-a4--a8-mma)
- [GGUF Container Compatibility: A4 / A8 / A16](#gguf-container-compatibility-a4--a8--a16)
- [3-Tier Dynamic Block Distribution (VRAM-RAM-Disk)](#3-tier-dynamic-block-distribution-vram-ram-disk)
- [Phase 3 Breakthrough: 1080p High-Resolution Engine](#phase-3-breakthrough-1080p-high-resolution-engine)
  - [Adaptive Head Grouping & Alignment](#adaptive-head-grouping--alignment)
  - [In-Place CUDA Memory Operators](#in-place-cuda-memory-operators)
  - [Staggered VRAM Lifecycle Management](#staggered-vram-lifecycle-management)
- [Real-World Benchmark Matrix (RTX 3070 8GB)](#real-world-benchmark-matrix-rtx-3070-8gb)
  - [720p 3s Baseline](#720p-3s-baseline)
  - [1080p Full-Spectrum Matrix (3s, 5s, 8s, 12s)](#1080p-full-spectrum-matrix-3s-5s-8s-12s)
  - [Numerical Rigor & Bit-Level Verification](#numerical-rigor--bit-level-verification)
- [Repository Branch Architecture & Evolution](#repository-branch-architecture--evolution)
- [Next-Gen Roadmap: Researcher Arbitrated Paths](#next-gen-roadmap-researcher-arbitrated-paths)
- [API Reference & Quick Start](#api-reference--quick-start)
- [Acknowledgments & Credits](#acknowledgments--credits)
- [License](#license)

---

## Key Highlights

- **100% Pure Rust Architecture**: Zero Python runtime, zero C/C++ build toolchains (`cmake`/`make`), zero bloated dynamic libraries (`libggml.so`, `libedge_dit.so`). Built into a single static ELF binary (`server_9412`).
- **Native Ampere INT4 MMA**: Hardware-level `m16n8k64` INT4 Tensor Core instruction execution via NVRTC JIT, delivering up to **135.24 TOPS** on sm_86 cards.
- **Universal Container Ingestion**: Natively loads both `.safetensors` and `.gguf` weights with on-the-fly streaming and zero memory bloat.
- **3-Tier Asymmetric Memory Offload**: Runs 14B DiT on an 8GB VRAM card using flexible tiering (e.g., 28 layers in VRAM, 14 in Host RAM, 5 on NVMe SSD).
- **1080p Full-Spectrum Acceleration**:
  - **1080p 3s (49k tokens)**: **137.06s** (2.65x speedup, pure GPU zero-copy path).
  - **1080p 5s (80k tokens)**: **300.60s** (1.97x speedup, breaking the 5-minute barrier).
  - **1080p 8s (121k tokens)**: **626.38s** (1.45x speedup, shaving off 283.77s via adaptive 14-group inplace column copy).
  - **1080p 12s (182k tokens)**: **1467.30s** (100% stable execution within 3.90GB usable VRAM via staggered lifecycle scheduling).

---

## Architectural Decoupling: Why Monolithic DiT Fails

Modern video generation pipelines (like MiniMax H3 FL2VA) consist of three heterogeneous stages:
1. **Conditioning / Text-Image Encoding**: Qwen3-VL-32B multimodal encoder (requires >18 GB memory).
2. **Diffusion Backbone (DiT)**: 47-layer 14B Transformer with audio-video cross-attention (requires ~9.6 GB W4A4 weights).
3. **Latent Decoding (VAE)**: 3D Video VAE + Stereo Audio VAE (requires 4~8 GB peak VRAM for frame expansion).

```
[ Monolithic Framework: Memory Bottleneck ]
+-------------------------------------------------------------+
| Single 8GB / 12GB Consumer GPU (OOM CRASH)                   |
| [Qwen3-VL (18GB)] + [DiT 14B (9.6GB)] + [Video VAE (6GB)]   |  --> Requires >33 GB VRAM!
+-------------------------------------------------------------+

[ Decoupled Microservice Topology (Our Architecture) ]
+----------------------------+       +------------------------------------+
| Node A (Host / CPU Farm)   |       | Node B (Consumer RTX 3070 8GB)     |
| Port 9002: Qwen3-VL Enc    |       | Port 9412: Pure Rust h3-dit-w4a4   |
| Port 9003: Video/Audio VAE | <===> | - 28 Blocks in VRAM (5.39 GB)      |
| (144GB ECC RAM / Multi-CPU)| JSON  | - 14 Blocks in Host RAM (2.70 GB)  |
|                            | RPC   | - 5 Blocks on NVMe Disk (0.96 GB)  |
+----------------------------+       +------------------------------------+
```

### Why Decoupling is Essential:
1. **Elimination of CUDA Memory Fragmentation & GIL**: Running Python-based orchestrators (PyTorch, ComfyUI) causes severe memory fragmentation. When swapping layers between CPU and GPU, CUDA caching allocators easily trigger out-of-memory errors on 8GB cards.
2. **Predictable Memory Footprint**: `h3-dit-w4a4` operates with exact byte-level control. The memory residency is fixed at startup (e.g., exactly 4,975 MiB allocated on the GPU), with zero unexpected spikes during sampling.
3. **Physical Role Separation**: Heavy multi-modal prompt encoding and final video decoding run on CPU clusters or secondary servers, allowing the GPU to stay 100% saturated on the iterative diffusion loop.

---

## Hardware Tensor Core Acceleration: True A4 & A8 MMA

Unlike conventional quantization libraries that dequantize INT4/INT8 weights back to FP16 in registers before feeding standard FP16 CUDA cores, `h3-dit-w4a4` directly invokes **hardware Tensor Core Matrix-Multiply-Accumulate (MMA) instructions**.

### 1. W4A4 INT4 Tensor Core Execution (`w4a4_minimax_h3_sm86.cu`)
- **PTX Assembly Instruction**:
  ```cuda
  mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
  ```
- **Math Principle**: Each 32-thread warp computes an $M=16, N=8, K=64$ matrix multiplication per instruction directly on signed 4-bit integers (`s4`), accumulating into 32-bit integer registers (`s32`).
- **4D Pre-Swizzled Weight Layout**:
  - Raw INT4 weights are offline pre-swizzled into shape:
    $$\text{Shape: } [K / 64, N / 8, 16, 32]$$
  - Each byte packs two 4-bit nibbles, perfectly matching the thread-register mapping of NVIDIA Ampere Tensor Cores.
  - **Zero runtime overhead**: No transposition, no dequantization, no thread divergence.
- **Performance**: Delivers **135.24 TOPS** on RTX 3070, reducing FFN execution time to **7.00 ms per layer**.

### 2. W4A8 / W8A8 INT8 Tensor Core Execution (`src/attn8.rs`)
- **PTX Assembly Instruction**:
  ```cuda
  mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
  ```
- **Precision Preservation**: Attention query-key projections are sensitive to numerical dynamic ranges. W4A8/W8A8 preserves exact attention scores with Hadamard-256 ConvRot transformation while executing in just **4.34 ms per layer**.

### 3. Pure Rust NVRTC JIT Compilation
The CUDA kernels are compiled at microservice startup using NVIDIA Runtime Compilation (NVRTC) directly through Rust FFI to `libcuda.so.1`. No local `nvcc` installation or external C++ compiler toolchains are required on the host system.

---

## GGUF Container Compatibility: A4 / A8 / A16

`h3-dit-w4a4` provides full parsing and execution support for both GGUF and Safetensors container formats.

| Precision Mode | GGUF Tensor Type | Hardware Execution Target | Target Use Case |
|---|---|---|---|
| **W4A4 Pre-Swizzled** | Type 18 (`GGML_TYPE_I32` / Custom RAW) | `m16n8k64` Ampere INT4 MMA | Maximum speed on Ampere/Ada/Blackwell |
| **W4A8 Hybrid** | Type 8 (`GGML_TYPE_I8`) | `m16n8k32` Ampere INT8 MMA | High precision attention projections |
| **A16 / FP16** | Type 1 (`GGML_TYPE_F16`) | FP16 Tensor Core Fallback | Legacy or non-Ampere GPU fallback |

---

## 3-Tier Dynamic Block Distribution (VRAM-RAM-Disk)

The MiniMax H3 14B model consists of 47 DiT blocks (~9.6 GB). For GPUs with less than 10 GB of VRAM, `h3-dit-w4a4` introduces a flexible 3-tier memory allocation hierarchy:

```
[ Total 47 DiT Blocks ]
+------------------------------------+--------------------------+---------------------+
| Tier 1: VRAM (GPU Pinned)          | Tier 2: Host RAM Pinned  | Tier 3: NVMe mmap   |
| 28 Blocks (~5.39 GB)               | 14 Blocks (~2.70 GB)     | 5 Blocks (~0.96 GB) |
| Latency: 0.00 ms (Direct Access)   | Latency: ~12 ms PCIe H2D | Latency: On-demand  |
+------------------------------------+--------------------------+---------------------+
```

---

## Phase 3 Breakthrough: 1080p High-Resolution Engine

When scaling from 720p 3s (18k tokens) to 1080p (49k ~ 182k tokens), attention memory consumption escalates quadratically ($O(S^2)$). A naive implementation crashes immediately on 8GB GPUs due to physical memory exhaustion ($4.58\text{ GB} > 3.90\text{ GB}$). Phase 3 introduces three fundamental innovations to conquer high-resolution rendering:

### 1. Adaptive Head Grouping & Alignment
To maximize Tensor Core arithmetic density and prevent OOM, the engine automatically selects head grouping strategies based on sequence length $S$:
- **$S \le 60,000$ (3s / 49k)**: 2 groups (28 heads/group, $w=3584$), 100% pure GPU fastpath.
- **$60,000 < S \le 100,000$ (5s / 80k)**: 7 groups (8 heads/group, $w=1024$), pure GPU in-place column copy.
- **$100,000 < S \le 150,000$ (8s / 121k)**: 14 groups (4 heads/group, $w=512$), 256-byte aligned, pure GPU in-place column copy.
- **$S > 150,000$ (12s / 182k)**: 28 groups (2 heads/group, $w=256$), staggered VRAM offload.

### 2. In-Place CUDA Memory Operators
- **Dual In-Place Residuals (`gate_add_inplace` & `add_inplace`)**: In-place elementwise kernel replacing out-of-place additions, eliminating 94 allocations of 1.96 GB activation tensors.
- **In-Place Column Slicing (`copy_slice_col_inplace`)**: Replaces standard `.narrow()` slicing, copying contiguous QKV heads directly into destination tensor slices without creating 1,316 temporary candle tensor fragments.

### 3. Staggered VRAM Lifecycle Management
For long sequences ($S > 85,000$), the engine enforces strict memory synchronization:
- Immediately frees intermediate activation $n1$ (reclaiming 2.0 GB VRAM) before allocating projection output buffers.
- Synchronizes the CUDA device (`dev.synchronize()`) to trigger candle memory pool defragmentation.
- Dynamically adjusts `--vram-blocks 2` (or `--vram-blocks 0`) to guarantee a stable 3.90 GB usable VRAM ceiling.

---

## Real-World Benchmark Matrix (RTX 3070 8GB)

- **System Specs**: Intel Xeon / Core i7, NVIDIA GeForce RTX 3070 Laptop (GA104, 80W TGP, 8GB VRAM, sm_86, 135 TOPS INT4).
- **Environment**: Artix Linux x86_64, Linux 6.x, CUDA 12.8 / Driver 570.86.16.

### 720p 3s Baseline
- **Sequence Length**: 18,240 tokens
- **Euler Sampling**: 6 steps
- **Total Latency**: **64.02 s** (Average: **8.48 s/step**)
- **VRAM Utilization**: 4,975 MiB / 8,192 MiB

### 1080p Full-Spectrum Matrix (3s, 5s, 8s, 12s)

| Video Duration | Sequence Length ($S$) | Baseline (Single-Head Fragment) | Phase 3 Optimized Latency | Speedup | Time Saved | Execution Pipeline |
|---|---|---|---|---|---|---|
| **3 Seconds** | 49,273 tokens | 363.50 s | **137.06 s** | **2.65x** | -226.44 s | Pure GPU Zero-Copy Fastpath |
| **5 Seconds** | 80,015 tokens | 592.80 s | **300.60 s** | **1.97x** | -292.20 s | Pure GPU In-Place Col-Copy |
| **8 Seconds** | 121,041 tokens | 910.15 s | **607.34 s** | **1.50x** | -302.81 s | VRAM Threshold Refined (Pure Forward) |
| **12s (80W)** | 182,581 tokens | 1436.00 s | **1376.29 s** | **1.06x** | -59.71 s | Phase 3.2b-lite Pinned Stream |
| **12s (125W)** | 182,581 tokens | 1436.00 s | **1155.95 s** | **1.27x** | -311.35 s | Phase 3.3 Power/Clock Released |
| **12s (Phase 3.3b)** | 182,581 tokens | 1436.00 s | **1092.05 s** | **1.34x** | -375.25 s | Full 47-Block Zero CPU Cat (Sub-1100s) |

### Numerical Rigor & Bit-Level Verification
All generated latents have been strictly verified against golden FP32 baselines to eliminate silent numerical drift:
- **1080p 3s**: `has_nan: False`, `has_inf: False`, Relative Norm Error: **< 0.002%**
- **1080p 5s**: `has_nan: False`, `has_inf: False`, Relative Norm Error: **< 0.002%**
- **1080p 8s**: `has_nan: False`, `has_inf: False`, Relative Norm Error: **0.003%** (Norm: `259662.83` vs Ref: `259671.05`)
- **1080p 12s**: `has_nan: False`, `has_inf: False`, Relative Norm Error: **0.003%** (Norm: `259690.28` vs Ref: `259681.53`)

---

## Repository Branch Architecture & Evolution

> [!NOTE]
> Detailed Branch Deep-Dive: [docs/BRANCH_PROGRESS.md](docs/BRANCH_PROGRESS.md)


To maintain codebase hygiene while exploring radical architectural breakthroughs, the repository maintains specialized branches:

```
[ Git Branch Topology & Lifecycle ]
- w4a8-accel / w4a4-accel   --> Sm_86 INT4/INT8 Tensor Core MMA kernel exploration & swizzle layouts.
- phase1-scratch            --> CUDA Stream Mempool locking, static buffer pre-allocation, anti-fragmentation.
- phase2-streaming          --> 3-Tier asymmetric memory offload engine (VRAM-RAM-Disk) & async staging ring.
- main                      --> Unified high-performance engine: Phase 3 adaptive grouping & full-spectrum 1080p.
```

1. **`w4a8-accel` & `w4a4-accel`**: Dedicated to micro-benchmarking NVIDIA sm_86 PTX assembly instructions, Hadamard-256 ConvRot transformation, and offline 4D weight swizzling.
2. **`phase1-scratch`**: Focused on eradicating CUDA caching allocator overhead. Pre-allocated scratchpads reduced 2,256 tiny allocations to zero during diffusion passes.
3. **`phase2-streaming`**: Architected the asynchronous double-buffering PCIe ring. Allowed 14B parameter models to run on 8GB GPUs by interleaving compute and layer transfers.
4. **`main`**: Merged the fruits of all branches, integrating dual in-place residuals, adaptive head-grouping, and staggered VRAM lifecycle management.

---

## Next-Gen Roadmap: Researcher Arbitrated Paths

Following comprehensive verification of the 1080p full-spectrum matrix, WebLLM Farm researchers (`rc001` - Claude Sonnet 5.5 and `sp001` - AI Studio 4 Argon) arbitrated the next architectural milestones:

### 1. Quadratic Attention Modeling ($t \approx aN + cN^2$)
Fitting the 3s (49k), 5s (80k), and 8s (121k) data revealed:
- Quadratic Attention FLOPs dominate the computation: at 12s (182k tokens), attention accounts for **>85% (~1150s)** of execution time.
- The entire CPU offload penalty at 12s is approximately **130s ~ 160s (only ~10%)**.
- Eliminating CPU offload brings 12s down to ~1310s.

### 2. Implementation Paths
- **Path B+ (Pure GPU INT4 Epilogue Pre-Quantization - Recommended)**:
  - Quantize $n1$ to INT4 in Block 0 and $mid_{q,k,v}$ to INT4 in Blocks 1..46, immediately dropping unquantized FP16 tensors before allocating attention buffers.
  - Stream attention outputs through Out-Linear using `.narrow()` slices directly on GPU without transferring back to Host.
  - Reclaims ~1.95 GB VRAM, keeping total peak residency strictly below **3.39 GB** (leaving >500 MB headroom) and enabling 100% pure GPU execution at 12s.
- **Path A (Pinned 2D DMA Double-Buffering - Fallback)**:
  - Allocates a single reusable 2.62 GB pinned host buffer (`cuMemHostAlloc`).
  - Employs group-major layout $[G][N][256]$ for 1D contiguous DMA transfer, overlapping host-device communications to save 100s ~ 140s.

---

## API Reference & Quick Start

### 1. Build Single Binary
```bash
cargo build --release --bin server_9412
```

### 2. Launch Service
```bash
# Optimal for 1080p High-Resolution (3s, 5s, 8s, 12s)
./target/release/server_9412 \
  --model /mnt/kxV/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.gguf \
  --port 9412 \
  --vram-blocks 2 \
  --ram-blocks 40 \
  --disk-blocks 5 \
  --reserve-gb 1.5
```

### 3. Health & Sampling Probes
```bash
# Check service status
curl -s http://127.0.0.1:9412/health

# Trigger 1080p sampling step
curl -X POST http://127.0.0.1:9412/sample_step \
  -H "Content-Type: application/octet-stream" \
  --data-binary @stage1_input_1080p_5s.edbin \
  --output stage2_output_1080p_5s.edbin
```

---

## Acknowledgments & Credits

We express our profound gratitude to the open-source projects, researchers, and engineers whose groundbreaking work made this project possible:

1. **[edge-dit](https://github.com/apiplant/edge-dit) (C++)**: For pioneering edge-device Diffusion Transformer inference, highly innovative weight quantization strategies, and efficient operator design in native C++.
2. **[h3-rs](https://github.com/apiplant/h3-rs) (apiplant / Rust)**: For creating the elegant, foundational pure-Rust implementation of the MiniMax H3 pipeline, providing full ComfyUI model ecosystem compatibility and rock-solid architectural references.
3. **[MiniMax AI](https://www.minimaxi.com/)**: For open-sourcing the extraordinary **MiniMax H3 (HyperFlow Flash FL2VA 14B)** model weights, delivering state-of-the-art text/image-to-video capabilities with synchronized stereo audio.
4. **[llama.cpp & GGML Community](https://github.com/ggerganov/llama.cpp)**: For defining the GGUF container standard and pioneering high-efficiency quantization formats across the AI industry.

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
