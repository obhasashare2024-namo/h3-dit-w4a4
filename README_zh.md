# h3-dit-w4a4 (中文说明)

[English](README.md) | [中文说明](README_zh.md) | [日本語ドキュメント](README_ja.md)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust: 100%](https://img.shields.io/badge/Rust-100%25-orange.svg)](src/)
[![Architecture: Ampere sm_86+](https://img.shields.io/badge/Arch-Ampere%20sm__86%2B-green.svg)](w4a4_minimax_h3_sm86.cu)
[![Format: GGUF & Safetensors](https://img.shields.io/badge/Format-GGUF%20%2F%20Safetensors-blueviolet.svg)](#gguf-容器多精度兼容性-a4--a8--a16)

基于 **100% 纯 Rust** 构建的 **MiniMax H3 (HyperFlow Flash 14B)** 视频扩散变换器 (DiT) 高性能微服务推理引擎。原生集成 **NVIDIA Ampere INT4 (W4A4) 与 INT8 (W4A8) 硬件级 Tensor Core 硬解加速**、**GGUF 容器全规格无损解析**，以及独创的 **VRAM-RAM-Disk 三级动态流式分层卸载技术**。

专为低显存消费级显卡（如 RTX 3070 8GB / RTX 4060Ti / RTX 3060 12GB）量身定制，彻底打破大显存对 14B 视频扩散模型的垄断。

---

## 目录导航

- [核心特性](#核心特性)
- [架构解耦必要性：为何传统单体 DiT 必然崩溃](#架构解耦必要性为何传统单体-dit-必然崩溃)
- [硬件张量核心硬解：真 A4 与 A8 MMA 指令](#硬件张量核心硬解真-a4-与-a8-mma-指令)
- [GGUF 容器多精度兼容性：A4 / A8 / A16](#gguf-容器多精度兼容性a4--a8--a16)
- [三级显存-内存-磁盘分层流式加载 (3-Tier Allocation)](#三级显存-内存-磁盘分层流式加载-3-tier-allocation)
- [真实实测基准 (RTX 3070 8GB)](#真实实测基准-rtx-3070-8gb)
- [API 接口与极速上手](#api-接口与极速上手)
- [致谢与开源传承](#致谢与开源传承)

---

## 核心特性

- **100% 纯 Rust 自研微服务**：零 Python 运行时、零 C/C++ 外部构建工具链（无需 `cmake`、无需 `make`）、零笨重动态链接库（不依赖 `libggml.so` 或 `libedge_dit.so`）。全流程编译为单一静态 ELF 二进制文件（`server_9412`）。
- **真 Ampere INT4 MMA 硬解**：通过 NVRTC JIT 运行时原生调用底层 `m16n8k64` 硬件指令，在 sm_86 架构上释放 **135.24 TOPS** 满血峰值算力。
- **GGUF 容器原生零拷贝直通**：原生支持直接读取 `.safetensors` 与 `.gguf` 容器头，张量流式直通 GPU 共享内存，彻底规避内存二次复制。
- **三级不对称分层卸载 (VRAM-RAM-Disk)**：允许自由分配显卡显存、主机内存与 NVMe 磁盘块（例如 28 层在显存、14 层在内存、5 层在 NVMe 磁盘），使 8GB 显存显卡轻松驾驭 14B 参数大模型。
- **单步 8.48s 极致吞吐**：在 RTX 3070 8GB 笔记本/桌面端上跑出单步 **8.48 秒**（相较传统 FP16/W8A8 提速超 3.5 倍）。

---

## 架构解耦必要性：为何传统单体 DiT 必然崩溃

传统的视频生成流水线（以 MiniMax H3 FL2VA 为例）由三个计算异构阶段组成：
1. **条件提示词编码 (Conditioning)**：Qwen3-VL-32B 多模态视觉语言大模型（需要 >18 GB 显存/内存）。
2. **扩散主干网络 (DiT Backbone)**：47 层 14B 参数 Transformer，包含音视频跨模态交叉注意力（W4A4 权重大约 9.6 GB）。
3. **隐空间解码 (Latent VAE)**：3D 视频 VAE + 立体声音频 VAE（膨胀解码需要 4~8 GB 显存）。

```
[ 传统单体框架（ComfyUI / PyTorch）致命瓶颈 ]
+-------------------------------------------------------------+
| 8GB / 12GB 消费级显卡（必然触发 CUDA OOM 爆显存闪退）          |
| [Qwen3-VL (18GB)] + [DiT 14B (9.6GB)] + [Video VAE (6GB)]   |  --> 累计需要 >33 GB 显存！
+-------------------------------------------------------------+

[ 现代化解耦微服务拓扑（本项目架构）]
+----------------------------+       +------------------------------------+
| 调度与存储节点 (局域网 28 号机) |       | 纯 Rust 算力节点 (29 号机 RTX 3070) |
| 端口 9002: Qwen3-VL 编码服务 |       | 端口 9412: h3-dit-w4a4 独立微服务   |
| 端口 9003: VAE 视频/音频解码 | <===> | - 28 层常驻显存 VRAM (5.39 GB)     |
| (144GB ECC RAM / 96 核 CPU) | JSON  | - 14 层常驻主机内存 RAM (2.70 GB)  |
|                            | RPC   | - 5 层置于高速 NVMe 磁盘 (0.96 GB)  |
+----------------------------+       +------------------------------------+
```

### 为什么必须坚决解耦？
1. **杜绝 Python GIL 锁与 PyTorch 显存碎片化**：Python 的全局解释器锁和 PyTorch 的 CUDA Caching Allocator 在高频层交换时会产生严重的内存碎片。在 8GB 显存的临界边缘，任何碎片都会直接导致不可逆的 OOM 崩溃。
2. **绝对可预测的物理内存驻留**：`h3-dit-w4a4` 具备微秒级的字节控制力。服务启动加载后，GPU 显存恒定保持在 4,975 MiB，采样过程中没有任何动态显存分配突刺。
3. **异构计算资源物尽其用**：庞大的 32B 文本编码和 VAE 解码交给内存充裕的 CPU 服务器或第二显卡处理；让专职显卡 100% 专注于高频的 DiT 扩散步迭代。

---

## 硬件张量核心硬解：真 A4 与 A8 MMA 指令

许多开源量化方案本质上是“假的 INT4”——它们在寄存器中先将 INT4 反量化解包为 FP16，然后再调用普通的 FP16 CUDA 核心计算，这不仅没有提升吞吐，反而增加了显存带宽与解包计算开销。`h3-dit-w4a4` 深入硬件底层，**直接调用 Ampere 原生 Tensor Core 矩阵乘累加 (MMA) 指令**。

### 1. W4A4 INT4 硬件核心 (`w4a4_minimax_h3_sm86.cu`)
- **PTX 汇编指令**：
  ```cuda
  mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
  ```
- **数学与硬件原理**：每个由 32 个线程组成的 Warp，单条指令直接对有符号 4 位整数 (`s4`) 完成 $M=16, N=8, K=64$ 的超大块矩阵乘累加，累加结果保存在 32 位整数寄存器 (`s32`) 中。
- **4D 预重排 (Pre-Swizzled) 权重排布**：
  - 权重在量化阶段离线进行硬件对齐排布：
    $$\text{维度结构: } [K / 64, N / 8, 16, 32]$$
  - 每个字节紧凑容纳两个 4-bit Nibble，物理排布与 Ampere Tensor Core 内部线程寄存器一一对应。
  - **零运行时开销**：运行时无需转置、无需解码、零分支分歧。
- **实测表现**：在 RTX 3070 上达成 **135.24 TOPS**，将 FFN 模块的单层运算延迟直接压缩至 **7.00 ms**。

### 2. W4A8 / W8A8 INT8 硬件核心 (`src/attn8.rs`)
- **PTX 汇编指令**：
  ```cuda
  mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
  ```
- **保真度与动态范围**：注意力层中的 Query-Key 点乘对动态范围极其敏感。我们使用 W4A8/W8A8 搭配 Hadamard-256 (ConvRot) 旋转变换，既消除了激活异常值（Outliers），又实现了单层 **4.34 ms** 的极限响应。

### 3. 纯 Rust NVRTC JIT 动态编译
CUDA 算子通过 Rust 原生绑定直接调用 NVIDIA 驱动层的 NVRTC (`libcuda.so.1`) 在服务拉起时即时编译为 PTX。宿主机无需安装庞大的 CUDA SDK 或编译器套件，二进制定型即用。

---

## GGUF 容器多精度兼容性：A4 / A8 / A16

本项目对 GGUF 与 Safetensors 容器提供了全方位的解析与调度支持：

| 精度模式 | GGUF 张量类型 | 硬件执行单元 | 适用场景 |
|---|---|---|---|
| **W4A4 Pre-Swizzled** | Type 18 (`GGML_TYPE_I32` / RAW) | `m16n8k64` Ampere INT4 MMA | Ampere/Ada/Blackwell 显卡极致加速 |
| **W4A8 混合精度** | Type 8 (`GGML_TYPE_I8`) | `m16n8k32` Ampere INT8 MMA | 超高保真度注意力投影 |
| **A16 / FP16 回退** | Type 1 (`GGML_TYPE_F16`) | FP16 Tensor Core 兼容模式 | 非 Ampere 架构或测试验证 |

### 核心区别：Ampere 预重排 GGUF vs. 传统 GGML Q4_0
- **传统 GGML Q4_0 (Type 2)**：按 32 个数值一组进行分块量化，每个块带有 16 位的 Scale 比例因子。这种格式必须在 CPU/GPU 线程中逐个反量化展开，无法满足硬件 Tensor Core 一次性吞入 64 个 INT4 的严苛寄存器几何排布。
- **Ampere 4D Pre-Swizzled GGUF**：元数据注入 `h3.quant.format = "ampere_w4a4_group256_preswizzled"`。文件中的字节流即为硬件所需的瓦片（Tiles），直接通过 DMA 映射至 GPU 显存，跳过一切反量化环节。

### 流式格式极速转换工具
通过我们提供的零内存流式转换脚本，可在 12 秒内将 9.6GB 的 safetensors 转换为标准的 GGUF：
```bash
python3 tools/stream_convert_to_gguf.py \
  --input /path/to/minimax_h3_w4a4_preswizzled.safetensors \
  --output /path/to/minimax_h3_w4a4_preswizzled.gguf
```

---

## 三级显存-内存-磁盘分层流式加载 (3-Tier Allocation)

MiniMax H3 14B 模型包含 47 个 DiT 模块，总权重大约 9.6 GB。针对 8GB 显卡，`h3-dit-w4a4` 提供了非对称三级调度策略：

```
[ 47 个 DiT 变换器计算块 ]
+------------------------------------+--------------------------+---------------------+
| 第一级：显存 (GPU VRAM 常驻)       | 第二级：主机内存 (Host RAM) | 第三级：NVMe 高速磁盘 |
| 28 层 (~5.39 GB)                   | 14 层 (~2.70 GB)         | 5 层 (~0.96 GB)     |
| 访问延时: 0.00 ms (原地显存直接计算) | 传输延时: ~12 ms PCIe H2D| 访问延时: 按需换入换出 |
+------------------------------------+--------------------------+---------------------+
```

### 启动配置参数：
```bash
./target/release/server_9412 \
  --model /models/minimax_h3_w4a4_preswizzled.gguf \
  --port 9412 \
  --vram-blocks 28 \
  --ram-blocks 14 \
  --disk-blocks 5
```
- **双缓冲乒乓流 (Double-Buffering Ping-Pong Streaming)**：计算引擎使用两个独立的 CUDA Stream（计算流与传输流）。当 GPU 正在计算第 $N$ 层时，传输流已在后台通过 PCIe 4.0 预取第 $N+1$ 层，彻底掩盖宿主到显卡的传输延迟。

---

## 真实实测基准 (RTX 3070 8GB)

- **测试环境**：Artix Linux x86_64, NVIDIA GeForce RTX 3070 8GB Laptop (sm_86, INT4 135 TOPS)。
- **测试负载**：3 秒 720p 视频生成采样（`stage1_input_3s.edbin`），6 步 Euler 采样器。
- **测试模型**：`minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.gguf` (10.3 GB)。
- **显存稳定值**：**4,975 MiB / 8,192 MiB**（显存安全余量 >3.2 GB，零爆显存风险）。

```
Step 2 耗时: 9.011 s
Step 3 耗时: 8.482 s
Step 4 耗时: 8.503 s
Step 5 耗时: 8.518 s
Step 6 耗时: 8.483 s
Step 7 耗时: 8.480 s
---------------------------------------
6 步 Euler 采样总耗时: 64.02 s (平均单步: 8.48 s/step)
输出张量维度: 视频 [1, 24, 22, 28, 48], 音频 [1, 32, 2, 122]
数值完整性核查: has_nan = False (完全正常无溢出)
```

---

## API 接口与极速上手

### 1. 编译纯 Rust 二进制
```bash
cargo build --release --bin server_9412
```

### 2. 启动微服务
```bash
./target/release/server_9412 \
  --model /mnt/kxV/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.gguf \
  --port 9412 \
  --vram-blocks 28 \
  --ram-blocks 14 \
  --disk-blocks 5
```

### 3. 健康检查与采样调用
```bash
# 查询服务健康状态与后端架构
curl -s http://127.0.0.1:9412/health

# 执行单步扩散采样
curl -X POST http://127.0.0.1:9412/sample_step \
  -H "Content-Type: application/octet-stream" \
  --data-binary @stage1_input_3s.edbin \
  --output stage2_output_3s.edbin
```

---

## 致谢与开源传承

本项目的诞生离不开开源社区先驱者的奠基与无私分享。我们在此向以下开源项目、研究团队与工程师致以最诚挚的敬意与感谢：

1. **[edge-dit](https://github.com/apiplant/edge-dit) (C++)**：感谢其在边缘端扩散模型轻量化推理、算子设计与量化思想上的杰出探索与 C++ 工业级实现，为本项目奠定了宝贵的算子理论基石。
2. **[h3-rs](https://github.com/apiplant/h3-rs) (apiplant / Rust)**：感谢其创造的优雅、纯净的 MiniMax H3 Rust 原生实现，提供了与 ComfyUI 生态完全无缝的模型兼容体系与卓越的系统架构设计。
3. **[MiniMax AI](https://www.minimaxi.com/)**：感谢 MiniMax 团队开源令人惊叹的 **MiniMax H3 (HyperFlow Flash FL2VA 14B)** 权重，让高质量音视频同步生成真正走向大众。
4. **[llama.cpp 与 GGML 社区](https://github.com/ggerganov/llama.cpp)**：感谢其确立的 GGUF 容器标准及开创性的模型量化生态。

---

## 开源许可证

本项目采用 Apache 2.0 开源许可证。详见 [LICENSE](LICENSE)。\n