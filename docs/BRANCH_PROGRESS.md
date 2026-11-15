# Branch Evolution & Main Progress Report

[English](BRANCH_PROGRESS.md) | [中文说明](BRANCH_PROGRESS_zh.md) | [日本語ドキュメント](BRANCH_PROGRESS_ja.md)

This document provides a comprehensive architectural deep-dive into the developmental trajectory of **`h3-dit-w4a4`**, detailing both the specialized research branches (`w4a4-accel`, `phase1-scratch`, `phase2-streaming`) and the flagship unified **`main`** branch.

---

## Table of Contents

- [Branch Topology & Mission Matrix](#branch-topology--mission-matrix)
- [Individual Branch Deep-Dives](#individual-branch-deep-dives)
  - [1. Branch: `w4a4-accel` & `w4a8-accel` (Hardware Operator Foundation)](#1-branch-w4a4-accel--w4a8-accel-hardware-operator-foundation)
  - [2. Branch: `phase1-scratch` (CUDA Mempool Defragmentation)](#2-branch-phase1-scratch-cuda-mempool-defragmentation)
  - [3. Branch: `phase2-streaming` (3-Tier Asymmetric Memory Offload)](#3-branch-phase2-streaming-3-tier-asymmetric-memory-offload)
- [Main Branch: The Unified Phase 3 Engine](#main-branch-the-unified-phase-3-engine)
  - [Adaptive Head Grouping ($O(S^2)$ Mitigation)](#adaptive-head-grouping-os2-mitigation)
  - [Dual In-Place Residual Operators](#dual-in-place-residual-operators)
  - [Staggered VRAM Lifecycle Management](#staggered-vram-lifecycle-management)
- [1080p Full-Spectrum Milestone Verification](#1080p-full-spectrum-milestone-verification)
- [Next-Gen Roadmap: Researcher Arbitrated Paths](#next-gen-roadmap-researcher-arbitrated-paths)

---

## Branch Topology & Mission Matrix

To conquer consumer GPU hardware limitations (RTX 3070 8GB / sm_86 / 80W) while maintaining clean code architecture, development was split into targeted research branches before converging into `main`:

```
                    [ upstream: h3-rs (apiplant) ]
                                  |
            +---------------------+---------------------+
            |                     |                     |
     [ w4a4-accel ]        [ phase1-scratch ]    [ phase2-streaming ]
  (Hardware INT4 MMA)     (Mempool & Scratch)   (3-Tier Offload Engine)
  - m16n8k64 PTX JIT      - Zero Fragmentation  - VRAM:2/RAM:40/Disk:5
  - 4D Pre-Swizzled       - Static Scratchpad   - PCIe Async Ping-Pong
  - Hadamard-256 ConvRot  - 2,256 Allocs -> 0   - 14B on 8GB Bootstrapped
            |                     |                     |
            +---------------------+---------------------+
                                  |
                                  v
                               [ main ]
                   (Phase 3 Production Engine)
                   - 1080p Full-Spectrum (3s, 5s, 8s, 12s)
                   - Dual In-Place Residuals (Zero 1.96GB copies)
                   - In-Place Column Slicing (`copy_slice_col_inplace`)
                   - Adaptive Head Grouping (56 -> 28 -> 14 -> 2)
                   - Staggered VRAM Lifecycle (Free n1, dev.sync)
```

---

## Individual Branch Deep-Dives

### 1. Branch: `w4a4-accel` & `w4a8-accel` (Hardware Operator Foundation)
- **Primary Objective**: Bypass standard CUDA FP16 dequantization bottlenecks by issuing native Ampere Tensor Core instructions directly from pure Rust.
- **Key Breakthroughs**:
  1. **NVRTC JIT Integration**: Directly links to `libcuda.so.1`, compiling PTX assembly at startup without requiring local `nvcc`.
  2. **`m16n8k64` Execution**: Warps execute $16 \times 8 \times 64$ signed 4-bit matrix multiplies directly into 32-bit accumulators, unleashing **135.24 TOPS** on RTX 3070.
  3. **4D Offline Swizzle**: Formatted weights into $[K/64, N/8, 16, 32]$ layout, perfectly matching hardware warp-lane register banks and eliminating register bank conflicts.
  4. **Hadamard-256 ConvRot**: Preserves dynamic numerical ranges during attention projections, preventing precision collapse.

### 2. Branch: `phase1-scratch` (CUDA Mempool Defragmentation)
- **Primary Objective**: Eradicate CUDA caching allocator thrashing and out-of-memory crashes caused by frequent dynamic allocations across 47 DiT layers and multi-step Euler sampling.
- **Key Breakthroughs**:
  1. **CUDA Stream Mempool Locking**: Locked the memory pool threshold to prevent operating system page reclamation.
  2. **Static Scratchpad Pools**: Replaced dynamic allocations with pre-allocated intermediate activation buffers.
  3. **Allocation Overhead Reduction**: Reduced intermediate memory allocations from **2,256 allocations per step down to 0**, stabilizing frametimes.

### 3. Branch: `phase2-streaming` (3-Tier Asymmetric Memory Offload)
- **Primary Objective**: Enable a 9.6 GB 14B parameter diffusion model to fit within consumer cards with $\le 8\text{ GB}$ VRAM.
- **Key Breakthroughs**:
  1. **3-Tier Hierarchy (VRAM-RAM-Disk)**:
     - Tier 1 (VRAM): 2 ~ 28 blocks pinned in GPU memory for instantaneous compute.
     - Tier 2 (Host RAM): 14 ~ 40 blocks pinned in system memory via PCIe 4.0 transfers (~12 ms latency).
     - Tier 3 (NVMe SSD): 5 blocks dynamically streamed from disk on-demand.
  2. **Double-Buffering Async Staging Ring**: Designed ping-pong CUDA streams where layer $N+1$ is concurrently prefetched over PCIe while layer $N$ is computed on Tensor Cores, hiding host-to-device transfer latency.

---

## Main Branch: The Unified Phase 3 Engine

The `main` branch merges all foundational breakthroughs while introducing novel high-resolution 1080p operators:

### 1. Adaptive Head Grouping ($O(S^2)$ Mitigation)
Standard attention scales quadratically with sequence length $S$. Naively computing 1080p attention leads to instant out-of-memory errors ($4.58\text{ GB} > 3.90\text{ GB}$). `main` implements runtime sequence detection:
- **$S \le 60,000$ (3s / 49k tokens)**: 2 groups (28 heads/group), 100% pure GPU fastpath.
- **$60,000 < S \le 100,000$ (5s / 80k tokens)**: 7 groups (8 heads/group), pure GPU in-place column copy.
- **$100,000 < S \le 150,000$ (8s / 121k tokens)**: 14 groups (4 heads/group), 256-byte aligned in-place column copy.
- **$S > 150,000$ (12s / 182k tokens)**: 28 groups (2 heads/group), staggered VRAM offload.

### 2. Dual In-Place Residual Operators
- **`gate_add_inplace` & `add_inplace`**: Replaced out-of-place tensor addition, eliminating **94 allocations of 1.96 GB activation tensors** across 47 layers.
- **`copy_slice_col_inplace`**: In-place GPU column slicing kernel replacing `.narrow()`, copying QKV heads directly into destination tensor slices without creating 1,316 candle tensor fragments.

### 3. Staggered VRAM Lifecycle Management
For sequence lengths exceeding 85k tokens, `main` orchestrates strict staggered memory reclamation:
- Drops the intermediate modulated tensor $n1$ before allocating attention projection buffers, reclaiming 2.0 GB VRAM.
- Triggers `dev.synchronize()` to eliminate candle caching allocator fragmentation.

---

## 1080p Full-Spectrum Milestone Verification

All milestones have been verified on an **NVIDIA GeForce RTX 3070 Laptop (80W, 8GB VRAM)**:

| Metric | 1080p 3s | 1080p 5s | 1080p 8s | 1080p 12s |
|---|---|---|---|---|
| **Sequence Length ($S$)** | 49,273 tokens | 80,015 tokens | 121,041 tokens | 182,581 tokens |
| **Baseline Latency** | 363.50 s | 592.80 s | 910.15 s | 1436.00 s |
| **Phase 3 Latency** | **137.06 s** | **300.60 s** | **626.38 s** | **1467.30 s** |
| **Speedup Ratio** | **2.65x** | **1.97x** | **1.45x** | **1.00x** (OOM Conquered) |
| **Physical Norm Error** | < 0.002% | < 0.002% | 0.003% | 0.003% |

---

## Next-Gen Roadmap: Researcher Arbitrated Paths

Following full-spectrum verification, WebLLM Farm researchers (`rc001` - Claude Sonnet 5.5 and `sp001` - AI Studio 4 Argon) arbitrated the next optimization phases:

1. **Quadratic Complexity Modeling**:
   - $t \approx aN + cN^2$: Attention FLOPs account for **>85% (~1150s)** of compute at 12s.
   - Host offloading accounts for only **130s ~ 160s (~10%)**.
2. **Implementation Directions**:
   - **Path B+ (INT4 Pre-Quantization Pure GPU Path)**: Pre-quantize $n1$ in Block 0 and $mid_{q,k,v}$ in Blocks 1..46 to INT4, dropping FP16 buffers early to keep peak VRAM under **3.39 GB**, achieving 100% GPU residency at 12s.
   - **Path A (Pinned 2D DMA)**: Single 2.62 GB pinned host buffer with group-major $[G][N][256]$ layout to hide host-device transfer latencies.
