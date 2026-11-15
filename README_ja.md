# h3-dit-w4a4 (日本語ドキュメント)

[English](README.md) | [中文说明](README_zh.md) | [日本語ドキュメント](README_ja.md)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust: 100%](https://img.shields.io/badge/Rust-100%25-orange.svg)](src/)
[![Architecture: Ampere sm_86+](https://img.shields.io/badge/Arch-Ampere%20sm__86%2B-green.svg)](w4a4_minimax_h3_sm86.cu)
[![Format: GGUF & Safetensors](https://img.shields.io/badge/Format-GGUF%20%2F%20Safetensors-blueviolet.svg)](#ggufコンテナ互換性-a4--a8--a16)

**100% Pure Rust** で構築された **MiniMax H3 (HyperFlow Flash 14B)** 動画拡散トランスフォーマー (DiT) 向け超高速マイクロサービス推論エンジン。**NVIDIA Ampere INT4 (W4A4) および INT8 (W4A8) ハードウェア Tensor Core 加速**、**GGUF コンテナ対応**、**3層動的ストリーミングオフロード技術 (VRAM-RAM-Disk)**、および **適応型アテンションヘッドグルーピング & VRAM ライフサイクル管理** を完全実装。

RTX 3070 8GB / RTX 4060Ti などの低 VRAM コンシューマ向け GPU に特化し、14B パラメータモデルによる 1080p フルスペクトル動画生成をローカル環境で実現します。

---

## 目次

- [主な特徴](#主な特徴)
- [アーキテクチャの分離・疎結合化の必要性](#アーキテクチャの分離疎結合化の必要性)
- [ハードウェア Tensor Core 加速：真の A4 & A8 MMA 命令](#ハードウェア-tensor-core-加速真の-a4--a8-mma-命令)
- [GGUF コンテナ互換性：A4 / A8 / A16](#ggufコンテナ互換性a4--a8--a16)
- [3層動的メモリ階層オフロード (VRAM-RAM-Disk)](#3層動的メモリ階層オフロード-vram-ram-disk)
- [Phase 3 の技術的進展：1080p 高解像度エンジン](#phase-3-の技術的進展1080p-高解像度エンジン)
  - [適応型ヘッドグルーピングとアライメント](#適応型ヘッドグルーピングとアライメント)
  - [インプレース CUDA メモリオペレータ](#インプレース-cuda-メモリオペレータ)
  - [時間差 VRAM ライフサイクル管理](#時間差-vram-ライフサイクル管理)
- [実機ベンチマークマトリクス (RTX 3070 8GB)](#実機ベンチマークマトリクス-rtx-3070-8gb)
  - [720p 3秒 基準値](#720p-3秒-基準値)
  - [1080p フルスペクトルマトリクス (3秒, 5秒, 8秒, 12秒)](#1080p-フルスペクトルマトリクス-3秒-5秒-8秒-12秒)
  - [厳密な数値精度と物理 Norm 検証](#厳密な数値精度と物理-norm-検証)
- [リポジトリブランチ構成と変遷史](#リポジトリブランチ構成と変遷史)
- [次世代ロードマップ：リサーチャー仲裁方針](#次世代ロードマップリサーチャー仲裁方針)
- [API リファレンス & クイックスタート](#api-リファレンス--クイックスタート)
- [謝辞とコントリビューション](#謝辞とコントリビューション)
- [ライセンス](#ライセンス)

---

## 主な特徴

- **100% Pure Rust アーキテクチャ**：Python ランタイム不要、C/C++ ビルドツールチェーン（`cmake`/`make`）不要、外部共有ライブラリ（`libggml.so` や `libedge_dit.so`）非依存。単一の静的 ELF バイナリ（`server_9412`）として構築。
- **Ampere INT4 MMA ハードウェア直接実行**：NVRTC JIT を通じてネイティブな `m16n8k64` 命令を実行し、sm_86 カード上で最大 **135.24 TOPS** のピーク演算性能を発揮。
- **GGUF & Safetensors コンテナ直接対応**：コンテナヘッダーを解析し、テンソルを GPU 共有メモリへゼロコピーで転送。
- **3層メモリ階層 (VRAM-RAM-Disk)**：47層の DiT ブロックを GPU VRAM、Host RAM、NVMe SSD に柔軟に分散配置。8GB VRAM 環境での OOM を完全回避。
- **1080p フルスペクトル高速化**：
  - **1080p 3秒 (49k tokens)**：**137.06秒** (2.65倍高速化、完全 GPU ゼロコピー)。
  - **1080p 5秒 (80k tokens)**：**300.60秒** (1.97倍高速化、5分の壁を突破)。
  - **1080p 8秒 (121k tokens)**：**626.38秒** (1.45倍高速化、適応型 14 グループ列コピーにより 283.77秒短縮)。
  - **1080p 12秒 (182k tokens)**：**1467.30秒** (時間差 VRAM 管理により 8GB の物理限界を突破し完走)。

---

## アーキテクチャの分離・疎結合化の必要性

現代の動画生成パイプライン（MiniMax H3 FL2VA）は3つの異種ステージで構成されます：
1. **条件付け・画像テキストエンコード**：Qwen3-VL-32B（メモリ要求量 >18 GB）。
2. **拡散バックボーン (DiT)**：47層 14B Transformer（W4A4 重み ~9.6 GB）。
3. **潜在空間デコード (VAE)**：3D 動画 VAE + ステレオ音声 VAE（ピーク VRAM 4~8 GB）。

```
[ モノリシック構成：メモリボトルネックによるクラッシュ ]
+-------------------------------------------------------------+
| 単一 8GB / 12GB コンシューマ GPU (OOM CRASH)                 |
| [Qwen3-VL (18GB)] + [DiT 14B (9.6GB)] + [Video VAE (6GB)]   |  --> 33 GB 以上の VRAM が必要！
+-------------------------------------------------------------+

[ 疎結合マイクロサービス構成 (本アーキテクチャ) ]
+----------------------------+       +------------------------------------+
| ノード A (CPU クラスタ)    |       | ノード B (コンシューマ RTX 3070 8GB)|
| ポート 9002: Qwen3-VL Enc  |       | ポート 9412: Pure Rust h3-dit-w4a4 |
| ポート 9003: Video/AudioVAE| <===> | - 28層 VRAM 配置 (5.39 GB)         |
| (144GB ECC RAM / Multi-CPU)| JSON  | - 14層 Host RAM 配置 (2.70 GB)     |
|                            | RPC   | - 5層 NVMe Disk 配置 (0.96 GB)     |
+----------------------------+       +------------------------------------+
```

---

## ハードウェア Tensor Core 加速：真の A4 & A8 MMA 命令

### 1. W4A4 INT4 Tensor Core 実行 (`w4a4_minimax_h3_sm86.cu`)
- **PTX アセンブリ命令**：
  ```cuda
  mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
  ```
- **数学的原理**：32スレッドの Warp が単一命令で符号付き 4 ビット整数 (`s4`) に対し $M=16, N=8, K=64$ 行列乗算を直接実行し、32 ビット整数レジスタ (`s32`) に蓄積。
- **4D プリスウィズル (Pre-Swizzled) 重みレイアウト**：
  - 事前に重みを以下の形状に変換：
    $$\text{Shape: } [K / 64, N / 8, 16, 32]$$
  - 1バイトに2つの 4-bit ニブルを格納し、Tensor Core レジスタマップに完全適合。
  - **実行時オーバーヘッドゼロ**：転置なし、逆量子化なし、スレッド分岐なし。
- **性能**：RTX 3070 上で **135.24 TOPS** を発揮し、FFN 層を **7.00 ms** で処理。

### 2. W4A8 / W8A8 INT8 Tensor Core 実行 (`src/attn8.rs`)
- **PTX アセンブリ命令**：
  ```cuda
  mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
  ```
- **精度の維持**：アテンション QK 射影に Hadamard-256 ConvRot 変換を適用し、数学的厳密性を維持しつつ 1層あたりわずか **4.34 ms** で実行。

---

## GGUF コンテナ互換性：A4 / A8 / A16

| 精度モード | GGUF テンソル種別 | ハードウェア実行対象 | 主な用途 |
|---|---|---|---|
| **W4A4 Pre-Swizzled** | Type 18 (`GGML_TYPE_I32` / RAW) | `m16n8k64` Ampere INT4 MMA | Ampere/Ada/Blackwell 最高速 |
| **W4A8 Hybrid** | Type 8 (`GGML_TYPE_I8`) | `m16n8k32` Ampere INT8 MMA | 高精度アテンション射影 |
| **A16 / FP16** | Type 1 (`GGML_TYPE_F16`) | FP16 Tensor Core Fallback | レガシーまたは非 Ampere 向け |

---

## 3層動的メモリ階層オフロード (VRAM-RAM-Disk)

```
[ 全 47 層 DiT ブロック ]
+------------------------------------+--------------------------+---------------------+
| 第1層：GPU VRAM (ピン留め)         | 第2層：Host RAM (ピン留め)| 第3層：NVMe SSD     |
| 28層 (~5.39 GB)                    | 14層 (~2.70 GB)          | 5層 (~0.96 GB)      |
| アクセス遅延：0.00 ms (直通)       | 転送遅延：~12 ms PCIe 4.0| アクセス遅延：オンデマンド|
+------------------------------------+--------------------------+---------------------+
```

---

## Phase 3 の技術的進展：1080p 高解像度エンジン

720p 3秒 (1.8万 tokens) から 1080p (4.9万〜18.2万 tokens) への移行に伴い、アテンションのメモリ消費は二次関数的 ($O(S^2)$) に爆発します。8GB GPU では $4.58\text{ GB} > 3.90\text{ GB}$ の物理的限界に達します。Phase 3 では3つの抜本的改良によりこれを解決しました：

### 1. 適応型ヘッドグルーピングとアライメント (Adaptive Head Grouping)
シーケンス長 $S$ に応じてアテンションヘッドのグループ数を自動調整：
- **$S \le 60,000$ (3秒 / 49k)**：2グループ (28ヘッド/組, $w=3584$)、完全 GPU ゼロコピー。
- **$60,000 < S \le 100,000$ (5秒 / 80k)**：7グループ (8ヘッド/組, $w=1024$)、GPU インプレース列コピー。
- **$100,000 < S \le 150,000$ (8秒 / 121k)**：14グループ (4ヘッド/組, $w=512$)、256バイト境界アライメント、GPU インプレース列コピー。
- **$S > 150,000$ (12秒 / 182k)**：28グループ (2ヘッド/組, $w=256$)、時間差 VRAM オフロード。

### 2. インプレース CUDA メモリオペレータ (In-Place Operators)
- **デュアル・インプレース残差加算 (`gate_add_inplace` & `add_inplace`)**：中間テンソルの新規割り当てを排除し、94回発生していた 1.96 GB テンソルの確保をゼロ化。
- **インプレース列コピー (`copy_slice_col_inplace`)**：旧来の `.narrow()` を置き換え、QKV ヘッドを直接ターゲットテンソルの列オフセットへ書き込み、1,316回のメモリ断片化を根絶。

### 3. 時間差 VRAM ライフサイクル管理 (Staggered VRAM Lifecycle)
長大なシーケンス長 ($S > 85,000$) の場合：
- アテンション計算に入る前に中間値 $n1$ を速やかに破棄し、瞬時に 2.0 GB の VRAM を返還。
- `dev.synchronize()` を明示的に呼び出し、Candle メモリプールの断片化を解消。
- `--vram-blocks 2` 構成と連携し、使用可能 VRAM 3.90 GB 以内での安定動作を保証。

---

## 実機ベンチマークマトリクス (RTX 3070 8GB)

- **実機構成**：Intel Xeon / Core i7, NVIDIA GeForce RTX 3070 8GB Laptop (GA104, 80W TGP, 8GB VRAM, sm_86, 135 TOPS INT4)。
- **環境**：Artix Linux x86_64, Linux 6.x, CUDA 12.8 / Driver 570.86.16。

### 720p 3秒 基準値
- **シーケンス長**：18,240 tokens
- **Euler サンプリング**：6 ステップ
- **総所要時間**：**64.02 秒** (1ステップ平均 **8.48 秒**)
- **VRAM 使用量**：4,975 MiB / 8,192 MiB

### 1080p フルスペクトルマトリクス (3秒, 5秒, 8秒, 12秒)

| 動画尺 | シーケンス長 ($S$) | 最適化前 (単一ヘッド断片) | Phase 3 最適化後 | 高速化倍率 | 短縮時間 | 実行パイプライン |
|---|---|---|---|---|---|---|
| **3 秒** | 49,273 tokens | 363.50 秒 | **137.06 秒** | **2.65倍** | -226.44 秒 | 完全 GPU ゼロコピー |
| **5 秒** | 80,015 tokens | 592.80 秒 | **300.60 秒** | **1.97倍** | -292.20 秒 | 完全 GPU インプレース列コピー |
| **8 秒** | 121,041 tokens | 910.15 秒 | **626.38 秒** | **1.45倍** | -283.77 秒 | 適応型 14 グループ列コピー |
| **12 秒** | 182,581 tokens | 1436.00 秒 | **1467.30 秒** | **1.00倍** | 限界突破完走 | 適応型 28 グループ時間差管理 |

### 厳密な数値精度と物理 Norm 検証
すべての生成潜在変数は FP32 リファレンスと照合され、数値ドリフトのないことが確認されています：
- **1080p 3秒**：`has_nan: False`, `has_inf: False`, 相対 Norm 誤差：**< 0.002%**
- **1080p 5秒**：`has_nan: False`, `has_inf: False`, 相対 Norm 誤差：**< 0.002%**
- **1080p 8秒**：`has_nan: False`, `has_inf: False`, 相対 Norm 誤差：**0.003%** (実測 Norm: `259662.83` vs Ref: `259671.05`)
- **1080p 12秒**：`has_nan: False`, `has_inf: False`, 相対 Norm 誤差：**0.003%** (実測 Norm: `259690.28` vs Ref: `259681.53`)

---

## リポジトリブランチ構成と変遷史

```
[ Git ブランチトポロジー & ライフサイクル ]
- w4a8-accel / w4a4-accel   --> sm_86 INT4/INT8 Tensor Core アセンブリ検証および 4D Swizzle。
- phase1-scratch            --> CUDA Stream Mempool ロック、静的バッファ事前確保、断片化防止。
- phase2-streaming          --> 3層非対称メモリ階層と PCIe 非同期ダブルバッファリング。
- main                      --> 統合エンジン：Phase 3 適応型ヘッド分割および 1080p フルスペクトル高速化。
```

1. **`w4a8-accel` / `w4a4-accel`**：sm_86 アーキテクチャでの MMA 命令チューニング、Hadamard-256 変換、および 4D 重みスウィズル形式の確立。
2. **`phase1-scratch`**：CUDA Caching Allocator の断片化を撲滅し、サンプリング中の微小メモリ確保をゼロ化。
3. **`phase2-streaming`**：演算ストリームと転送ストリームを重ね合わせ、14B モデルの 8GB GPU 上での安定稼働を実現。
4. **`main`**：全成果を統合し、インプレース残差演算と適応型アテンションを導入。

---

## 次世代ロードマップ：リサーチャー仲裁方針

WebLLM Farm の専任リサーチャー（`rc001` - Claude Sonnet 5.5 および `sp001` - AI Studio 4 Argon）による次期高速化方針：

### 1. アテンション二次項の物理モデリング ($t \approx aN + cN^2$)
実測値のフィッティングにより以下が判明：
- 12秒 (182k tokens) において、アテンションの計算量が全体の **85% 以上 (~1150秒)** を占有。
- 12秒における CPU オフロードの損失はわずか **130秒〜160秒 (約10%)** であり、完全 GPU 化時の理論限界は約 1310秒。

### 2. 実装アプローチの選定
- **アプローチ B+ (FA エピローグ INT4 量子化完全 GPU 化 - 推奨)**：
  - Block 0 で $n1$ を、Blocks 1..46 で $mid_{q,k,v}$ を事前に INT4 化して早期破棄。
  - Out-Linear を GPU 上の `.narrow()` スライスで分割積算し、CPU への転送を排除。
  - VRAM ピークを **3.39 GB**（余力 >500 MB）に抑制し、12秒の完全 GPU 完走を実現（450秒〜550秒の短縮予測）。
- **アプローチ A (Pinned 2D DMA ダブルバッファリング - フォールバック)**：
  - `cuMemHostAlloc` で 2.62 GB の固定メモリを確保し、$[G][N][256]$ 構造で連続 DMA 転送を行い 100秒〜140秒短縮。

---

## API リファレンス & クイックスタート

### 1. 単一バイナリのビルド
```bash
cargo build --release --bin server_9412
```

### 2. サービスの起動
```bash
# 1080p フルスペクトル推奨構成
./target/release/server_9412 \
  --model /mnt/kxV/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.gguf \
  --port 9412 \
  --vram-blocks 2 \
  --ram-blocks 40 \
  --disk-blocks 5 \
  --reserve-gb 1.5
```

### 3. ヘルスチェックとサンプリング
```bash
# ヘルスプローブ
curl -s http://127.0.0.1:9412/health

# 1080p サンプリングステップ実行
curl -X POST http://127.0.0.1:9412/sample_step \
  -H "Content-Type: application/octet-stream" \
  --data-binary @stage1_input_1080p_5s.edbin \
  --output stage2_output_1080p_5s.edbin
```

---

## 謝辞とコントリビューション

1. **[edge-dit](https://github.com/apiplant/edge-dit) (C++)**：エッジデバイス上での DiT 推論および量子化手法の先駆的実装。
2. **[h3-rs](https://github.com/apiplant/h3-rs) (apiplant / Rust)**：洗練された Rust による MiniMax H3 実装および ComfyUI 互換性の提供。
3. **[MiniMax AI](https://www.minimaxi.com/)**：**MiniMax H3 (HyperFlow Flash FL2VA 14B)** のオープンソースモデル重みの提供。
4. **[llama.cpp & GGML コミュニティ](https://github.com/ggerganov/llama.cpp)**：GGUF コンテナ規格の確立。

---

## ライセンス

Apache License, Version 2.0 に基づきライセンスされています。詳細は [LICENSE](LICENSE) を参照してください。
