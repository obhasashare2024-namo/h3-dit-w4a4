# h3-dit-w4a4

[English](README.md) | [中文说明](README_zh.md) | [日本語ドキュメント](README_ja.md)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust: 100%](https://img.shields.io/badge/Rust-100%25-orange.svg)](src/)
[![Architecture: Ampere sm_86+](https://img.shields.io/badge/Arch-Ampere%20sm__86%2B-green.svg)](w4a4_minimax_h3_sm86.cu)
[![Format: GGUF & Safetensors](https://img.shields.io/badge/Format-GGUF%20%2F%20Safetensors-blueviolet.svg)](#gguf-container-compatibility-a4--a8--a16)

High-performance, pure Rust microservice engine for **MiniMax H3 (HyperFlow Flash 14B)** text/image-to-video DiT (Diffusion Transformer) inference, powered by **native NVIDIA Ampere Tensor Core INT4 (W4A4) and INT8 (W4A8) hardware acceleration**, **GGUF container compatibility**, and **3-Tier (VRAM-RAM-Disk) dynamic memory streaming**.

Engineered specifically for low-VRAM consumer GPUs (e.g., NVIDIA GeForce RTX 3070 8GB / RTX 4060Ti / A4000), decoupling heavy diffusion compute from monolithic runtimes.

---

## Table of Contents

- [Key Highlights](#key-highlights)
- [Architectural Decoupling: Why Monolithic DiT Fails](#architectural-decoupling-why-monolithic-dit-fails)
- [Hardware Tensor Core Acceleration: True A4 & A8 MMA](#hardware-tensor-core-acceleration-true-a4--a8-mma)
- [GGUF Container Compatibility: A4 / A8 / A16](#gguf-container-compatibility-a4--a8--a16)
- [3-Tier Dynamic Block Distribution (VRAM-RAM-Disk)](#3-tier-dynamic-block-distribution-vram-ram-disk)
- [Real-World Benchmark (RTX 3070 8GB)](#real-world-benchmark-rtx-3070-8gb)
- [API Reference & Quick Start](#api-reference--quick-start)
- [Acknowledgments & Credits](#acknowledgments--credits)

---

## Key Highlights

- **100% Pure Rust Architecture**: Zero Python runtime, zero C/C++ build toolchains (`cmake`/`make`), zero bloated dynamic libraries (`libggml.so`, `libedge_dit.so`). Built into a single static ELF binary (`server_9412`).
- **Native Ampere INT4 MMA**: Hardware-level `m16n8k64` INT4 Tensor Core instruction execution via NVRTC JIT, delivering up to **135.24 TOPS** on sm_86 cards.
- **Universal Container Ingestion**: Natively loads both `.safetensors` and `.gguf` weights with on-the-fly streaming and zero memory bloat.
- **3-Tier Asymmetric Memory Offload**: Runs 14B DiT on an 8GB VRAM card using flexible tiering (e.g., 28 layers in VRAM, 14 in Host RAM, 5 on NVMe SSD).
- **Sub-9s Step Latency**: Achieves **8.48s per Euler step** on consumer-grade RTX 3070 8GB (down from 28s+ in unoptimized FP16/W8A8).

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

### Contrast: Our Ampere Pre-Swizzled GGUF vs. Legacy GGML Q4_0
- **Legacy GGML Q4_0 (Type 2)**: Formatted in 32-value blocks with 16-bit block scales. Requires dequantization threads on CPU or GPU to unpack nibbles before arithmetic operations. It cannot directly feed hardware Tensor Core MMA `k64` matrix registers.
- **Ampere 4D Pre-Swizzled GGUF**: Stores tensors with `h3.quant.format = "ampere_w4a4_group256_preswizzled"`. Bytes map directly to Tensor Core register tiles, enabling pure DMA copy from disk/RAM directly into GPU shared memory.

### Fast Stream-Conversion (Safetensors to GGUF)
You can convert safetensors into pre-swizzled GGUF in seconds with our zero-RAM stream converter:
```bash
python3 tools/stream_convert_to_gguf.py \
  --input /path/to/minimax_h3_w4a4_preswizzled.safetensors \
  --output /path/to/minimax_h3_w4a4_preswizzled.gguf
# Converts a 9.6 GB model in ~11.3 seconds
```

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

### Memory Allocation Flags:
```bash
./target/release/server_9412 \
  --model /models/minimax_h3_w4a4_preswizzled.gguf \
  --port 9412 \
  --vram-blocks 28 \
  --ram-blocks 14 \
  --disk-blocks 5
```
- **Double-Buffering Ping-Pong Streaming**: The compute engine alternates between two CUDA streams (Compute Stream and H2D Transfer Stream). While the GPU computes Block $N$, Block $N+1$ is concurrently prefetched across PCIe 4.0, hiding host-to-device transfer latency.

---

## Real-World Benchmark (RTX 3070 8GB)

- **System**: Artix Linux x86_64, Intel Xeon / Core i7, NVIDIA GeForce RTX 3070 8GB Laptop/Desktop (GA104, sm_86, 135 TOPS INT4).
- **Workload**: 3-second 720p video generation (`stage1_input_3s.edbin`), 6 Euler sampling steps.
- **Model**: `minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.gguf` (10.3 GB).
- **VRAM Utilization**: Stable at **4,975 MiB / 8,192 MiB** (Zero OOM).

```
Step 2: 9.011 s
Step 3: 8.482 s
Step 4: 8.503 s
Step 5: 8.518 s
Step 6: 8.483 s
Step 7: 8.480 s
---------------------------------------
Total 6-Step Latency: 64.02 s (Avg: 8.48 s/step)
Output Tensors: Video [1, 24, 22, 28, 48], Audio [1, 32, 2, 122]
Numerical Verification: has_nan = False
```

---

## API Reference & Quick Start

### 1. Build Single Binary
```bash
cargo build --release --bin server_9412
```

### 2. Launch Service
```bash
./target/release/server_9412 \
  --model /mnt/kxV/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.gguf \
  --port 9412 \
  --vram-blocks 28 \
  --ram-blocks 14 \
  --disk-blocks 5
```

### 3. Health & Sampling Probes
```bash
# Check service status
curl -s http://127.0.0.1:9412/health

# Trigger sampling step
curl -X POST http://127.0.0.1:9412/sample_step \
  -H "Content-Type: application/octet-stream" \
  --data-binary @stage1_input_3s.edbin \
  --output stage2_output_3s.edbin
```

---

## Acknowledgments & Credits

We stand on the shoulders of giants. We express our profound gratitude to the open-source projects, researchers, and engineers whose groundbreaking work made this project possible:

1. **[edge-dit](https://github.com/apiplant/edge-dit) (C++)**: For pioneering edge-device Diffusion Transformer inference, highly innovative weight quantization strategies, and efficient operator design in native C++.
2. **[h3-rs](https://github.com/apiplant/h3-rs) (apiplant / Rust)**: For creating the elegant, foundational pure-Rust implementation of the MiniMax H3 pipeline, providing full ComfyUI model ecosystem compatibility and rock-solid architectural references.
3. **[MiniMax AI](https://www.minimaxi.com/)**: For open-sourcing the extraordinary **MiniMax H3 (HyperFlow Flash FL2VA 14B)** model weights, delivering state-of-the-art text/image-to-video capabilities with synchronized stereo audio.
4. **[llama.cpp & GGML Community](https://github.com/ggerganov/llama.cpp)**: For defining the GGUF container standard and pioneering high-efficiency quantization formats across the AI industry.

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.\n