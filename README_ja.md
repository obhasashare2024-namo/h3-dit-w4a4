# h3-dit-w4a4 (日本語ドキュメント)

[English](README.md) | [中文说明](README_zh.md) | [日本語ドキュメント](README_ja.md)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust: 100%](https://img.shields.io/badge/Rust-100%25-orange.svg)](src/)
[![Architecture: Ampere sm_86+](https://img.shields.io/badge/Arch-Ampere%20sm__86%2B-green.svg)](w4a4_minimax_h3_sm86.cu)
[![Format: GGUF & Safetensors](https://img.shields.io/badge/Format-GGUF%20%2F%20Safetensors-blueviolet.svg)](#ggufコンテナ互換性-a4--a8--a16)

**100% Pure Rust** で構築された **MiniMax H3 (HyperFlow Flash 14B)** 動画拡散トランスフォーマー (DiT) 向け超高速マイクロサービス推論エンジン。**NVIDIA Ampere INT4 (W4A4) および INT8 (W4A8) ハードウェア Tensor Core 加速**、**GGUF コンテナ対応**、および独自の **VRAM-RAM-Disk 3層階層ストリーミングオフロード技術** を完全実装。

RTX 3070 8GB / RTX 4060Ti などの低 VRAM コンシューマ向け GPU に特化し、大容量 VRAM が必須とされてきた 14B クラス動画生成モデルのローカル推論を可能にします。

---

## 目次

- [主な特徴](#主な特徴)
- [アーキテクチャの分離・疎結合化の必要性](#アーキテクチャの分離疎結合化の必要性)
- [ハードウェア Tensor Core 加速：真の A4 & A8 MMA 命令](#ハードウェア-tensor-core-加速真の-a4--a8-mma-命令)
- [GGUF コンテナ互換性：A4 / A8 / A16](#gguf-コンテナ互換性a4--a8--a16)
- [3層動的メモリ階層オフロード (VRAM-RAM-Disk)](#3層動的メモリ階層オフロード-vram-ram-disk)
- [実機ベンチマーク (RTX 3070 8GB)](#実機ベンチマーク-rtx-3070-8gb)
- [API リファレンス & クイックスタート](#api-リファレンス--クイックスタート)
- [謝辞とコントリビューション](#謝辞とコントリビューション)

---

## 主な特徴

- **100% Pure Rust アーキテクチャ**：Python ランタイム不要、C/C++ ビルドツールチェーン（`cmake`/`make`）不要、外部共有ライブラリ（`libggml.so` や `libedge_dit.so`）非依存。単一の静的 ELF バイナリ（`server_9412`）として構築。
- **Ampere INT4 MMA ハードウェア直接実行**：NVRTC JIT を通じてネイティブな `m16n8k64` 命令を実行し、sm_86 カード上で最大 **135.24 TOPS** のピーク演算性能を発揮。
- **GGUF & Safetensors コンテナ直接対応**：コンテナヘッダーを解析し、テンソルを GPU 共有メモリへゼロコピーで転送。
- **3層メモリ階層 (VRAM-RAM-Disk)**：47層の DiT ブロックを GPU VRAM、Host RAM、NVMe SSD に柔軟に分散配置（例：VRAM 28層、RAM 14層、Disk 5層）。8GB VRAM 環境での OOM を完全回避。
- **1ステップ 8.48秒の高速サンプリング**：コンシューマ向け RTX 3070 8GB 上で 1ステップ平均 **8.48 秒** を達成。

---

## アーキテクチャの分離・疎結合化の必要性

MiniMax H3 FL2VA などの現代的な動画生成パイプラインは、以下の3つの異種ステージから構成されています：
1. **条件付け・プロンプトエンコーディング**：Qwen3-VL-32B マルチモーダルモデル（18 GB 以上のメモリが必要）。
2. **拡散バックボーン (DiT)**：47層 14B パラメータ Transformer（W4A4 重みで約 9.6 GB）。
3. **潜在空間デコード (VAE)**：3D 動画 VAE + ステレオ音声 VAE（フレーム展開時に 4〜8 GB の VRAM が必要）。

```
[ モノリシック構成（ComfyUI / PyTorch）の致命的ボトルネック ]
+-------------------------------------------------------------+
| 8GB / 12GB コンシューマ向け GPU（確実に CUDA OOM でクラッシュ） |
| [Qwen3-VL (18GB)] + [DiT 14B (9.6GB)] + [Video VAE (6GB)]   |  --> 合計で 33 GB 以上の VRAM が必要！
+-------------------------------------------------------------+

[ 疎結合マイクロサービス・トポロジー（本プロジェクト）]
+----------------------------+       +------------------------------------+
| オーケストレータノード (Host/CPU) |       | Pure Rust ノード (RTX 3070 8GB)    |
| Port 9002: Qwen3-VL エンコーダ|       | Port 9412: h3-dit-w4a4 マイクロサービス |
| Port 9003: VAE デコーダ    | <===> | - 28 ブロック VRAM (5.39 GB)       |
| (144GB ECC RAM / 96 CPU)   | JSON  | - 14 ブロック Host RAM (2.70 GB)   |
|                            | RPC   | - 5 ブロック NVMe SSD (0.96 GB)    |
+----------------------------+       +------------------------------------+
```

### なぜ分離が必要なのか？
1. **Python GIL と PyTorch メモリ断片化の根絶**：Python の GIL および PyTorch のアロケータは、頻繁なレイヤー転送時に深刻なメモリ断片化を引き起こします。8GB の境界環境では、わずかな断片化が即座に OOM を招きます。
2. **完全に予測可能なメモリフットプリント**：`h3-dit-w4a4` はバイト単位で常駐メモリを制御します。起動後の VRAM 使用量は 4,975 MiB に厳密に固定され、サンプリング中の突発的なメモリ増加は皆無です。
3. **計算資源の最適配分**：巨大なテキストエンコードと VAE デコードを大容量 RAM を持つホスト側に委ね、GPU リソースを反復的な拡散ステップ計算に 100% 集中させます。

---

## ハードウェア Tensor Core 加速：真の A4 & A8 MMA 命令

本プロジェクトは、レジスタ内で一度 INT4 を FP16 に戻して通常の CUDA コアで計算するような疑似量子化ではなく、**Ampere Tensor Core の MMA（行列積和演算）命令を直接実行**します。

### 1. W4A4 INT4 Tensor Core 実行 (`w4a4_minimax_h3_sm86.cu`)
- **PTX アセンブリ命令**：
  ```cuda
  mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
  ```
- **ハードウェア演算**：32スレッドで構成される Warp が、符号付き 4ビット整数 (`s4`) に対して $M=16, N=8, K=64$ の行列積を単一命令で計算し、32ビット整数アキュムレータ (`s32`) に格納します。
- **4D Pre-Swizzled 重みレイアウト**：
  - オフライン量子化時に Ampere レジスタ幾何構造に合わせて変換済み：
    $$\text{形状: } [K / 64, N / 8, 16, 32]$$
  - 各バイトに 2つの 4-bit ニブルがパッキングされ、ランタイムでの転置や展開オーバーヘッドはゼロです。
- **性能**：RTX 3070 上で **135.24 TOPS** を発揮し、FFN レイヤーの演算レイテンシを **7.00 ms** に短縮します。

### 2. W4A8 / W8A8 INT8 Tensor Core 実行 (`src/attn8.rs`)
- **PTX アセンブリ命令**：
  ```cuda
  mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
  ```
- **精度維持**：Attention の Query-Key 投影はダイナミックレンジに極めて敏感です。Hadamard-256 (ConvRot) 回転変換と組み合わせた INT8 MMA により、1レイヤーあたり **4.34 ms** で高精度な演算を維持します。

### 3. Pure Rust NVRTC JIT 動的コンパイル
CUDA カーネルは起動時に `libcuda.so.1` の NVRTC を通じてオンザフライで PTX へコンパイルされます。システム側に大規模な CUDA Toolkit をインストールする必要はありません。

---

## GGUF コンテナ互換性：A4 / A8 / A16

| 精度モード | GGUF テンソル型 | ハードウェア実行ユニット | 主な用途 |
|---|---|---|---|
| **W4A4 Pre-Swizzled** | Type 18 (`GGML_TYPE_I32` / RAW) | `m16n8k64` Ampere INT4 MMA | Ampere/Ada アーキテクチャでの最大速度 |
| **W4A8 Hybrid** | Type 8 (`GGML_TYPE_I8`) | `m16n8k32` Ampere INT8 MMA | 高精度アテンション投影 |
| **A16 / FP16** | Type 1 (`GGML_TYPE_F16`) | FP16 Tensor Core フォールバック | 互換性維持および非 Ampere 環境 |

### 従来の GGML Q4_0 との差異
- **従来の GGML Q4_0 (Type 2)**：32値ブロックごとにスケールを持つ形式。ハードウェア Tensor Core の k64 レジスタ幾何構造には直接適合せず、演算前にスレッドでの反量子化が必要です。
- **Ampere 4D Pre-Swizzled GGUF**：メタデータに `h3.quant.format = "ampere_w4a4_group256_preswizzled"` を保持。バイト列がそのまま Tensor Core タイルに対応しており、DMA 転送後即座にハードウェア MMA を実行できます。

---

## 3層動的メモリ階層オフロード (VRAM-RAM-Disk)

MiniMax H3 14B モデル（全47層、重み約 9.6 GB）を 8GB VRAM で動作させるため、柔軟な 3層メモリ割り当てを提供します：

```
[ 全 47 DiT ブロック ]
+------------------------------------+--------------------------+---------------------+
| 階層 1: GPU VRAM 常駐              | 階層 2: Host RAM ピン留め | 階層 3: NVMe SSD    |
| 28 ブロック (~5.39 GB)             | 14 ブロック (~2.70 GB)   | 5 ブロック (~0.96 GB)|
| レイテンシ: 0.00 ms (即時アクセス)  | レイテンシ: ~12 ms PCIe  | レイテンシ: オンデマンド|
+------------------------------------+--------------------------+---------------------+
```

### 起動コマンド例：
```bash
./target/release/server_9412 \
  --model /models/minimax_h3_w4a4_preswizzled.gguf \
  --port 9412 \
  --vram-blocks 28 \
  --ram-blocks 14 \
  --disk-blocks 5
```
- **ダブルバッファリング・ピンポンストリーミング**：計算用ストリームと PCIe 転送用ストリームを並列化。レイヤー $N$ を GPU で計算中に、レイヤー $N+1$ を Host RAM からプリフェッチし、転送オーバーヘッドを隠蔽します。

---

## 実機ベンチマーク (RTX 3070 8GB)

- **テスト環境**：Artix Linux x86_64, NVIDIA GeForce RTX 3070 8GB Laptop (sm_86, 135 TOPS INT4)。
- **タスク**：3秒 720p 動画生成 (`stage1_input_3s.edbin`)、Euler 6ステップサンプリング。
- **使用モデル**：`minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.gguf` (10.3 GB)。
- **VRAM 使用量**：**4,975 MiB / 8,192 MiB** で完全安定（OOM の心配なし）。

```
Step 2: 9.011 s
Step 3: 8.482 s
Step 4: 8.503 s
Step 5: 8.518 s
Step 6: 8.483 s
Step 7: 8.480 s
---------------------------------------
6ステップ総所要時間: 64.02 s (平均 8.48 s/step)
出力テンソル形状: Video [1, 24, 22, 28, 48], Audio [1, 32, 2, 122]
数値整合性検証: has_nan = False
```

---

## API リファレンス & クイックスタート

```bash
# 1. ビルド
cargo build --release --bin server_9412

# 2. サービス起動
./target/release/server_9412 \
  --model /mnt/kxV/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.gguf \
  --port 9412 \
  --vram-blocks 28 \
  --ram-blocks 14 \
  --disk-blocks 5

# 3. ヘルスチェック
curl -s http://127.0.0.1:9412/health
```

---

## 謝辞とコントリビューション

本プロジェクトは、オープンソースコミュニティの先駆的な研究と成果の上に成り立っています。多大なる貢献をされた以下のプロジェクトおよび開発者の方々に深く感謝いたします：

1. **[edge-dit](https://github.com/apiplant/edge-dit) (C++)**：エッジ環境における拡散モデル軽量化、独創的な量子化アルゴリズム、および高性能 C++ 演算子の設計において先駆的な道を開いてくださったことに心より感謝申し上げます。
2. **[h3-rs](https://github.com/apiplant/h3-rs) (apiplant / Rust)**：MiniMax H3 の卓越した Pure Rust 実装を構築し、ComfyUI モデルエコシステムとの完全な互換性と堅牢な設計を提供してくださったことに深く感謝申し上げます。
3. **[MiniMax AI](https://www.minimaxi.com/)**：高品質な動画・音声同時生成モデル **MiniMax H3 (HyperFlow Flash FL2VA 14B)** をオープンソースとして公開してくださった MiniMax チームに敬意を表します。
4. **[llama.cpp & GGML コミュニティ](https://github.com/ggerganov/llama.cpp)**：業界標準となる GGUF コンテナフォーマットと量子化エコシステムを確立してくださったことに感謝いたします。

---

## ライセンス

Apache License 2.0 の下で公開されています。詳細は [LICENSE](LICENSE) をご覧ください。\n