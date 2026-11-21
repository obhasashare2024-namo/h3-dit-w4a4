// ============================================================================
// src/dit.rs (或 src/radial_mask.rs)
// Version 6.3 P0 預備：3D 塊稀疏注意力 RadialAttentionMask 數據結構與掩碼生成器
// ============================================================================

use candle_core::{bail, Device, Result, Tensor};

/// 音頻 / 文本條件 Token 在序列中的排列位置
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CondTokenLayout {
    /// 條件 Token 位於序列尾端：[0 .. T*H*W) 為視頻，[T*H*W .. S) 為音頻/文本
    Suffix,
    /// 條件 Token 位於序列開頭：[0 .. cond_tokens) 為音頻/文本，[cond_tokens .. S) 為視頻
    Prefix,
}

/// 單一 Token 的 3D 時空座標語義
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenCoord3D {
    Video { t: u16, h: u16, w: u16 },
    Cond { cond_idx: u32 },
}

/// 3D 視頻潛空間網格幾何定義
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoGrid3D {
    pub t: usize,
    pub h: usize,
    pub w: usize,
    pub cond_tokens: usize,
    pub layout: CondTokenLayout,
}

impl VideoGrid3D {
    /// MiniMax H3 1088P 8s 實體尺度預設：
    /// Video = 25 * 68 * 71 = 120,700 tokens；Audio/Text = 341 tokens；Total S = 121,041
    pub const fn h3_1088p_8s(layout: CondTokenLayout) -> Self {
        Self {
            t: 25,
            h: 68,
            w: 71,
            cond_tokens: 341,
            layout,
        }
    }

    #[inline]
    pub const fn video_tokens(&self) -> usize {
        self.t * self.h * self.w
    }

    #[inline]
    pub const fn total_tokens(&self) -> usize {
        self.video_tokens() + self.cond_tokens
    }

    /// 將一維序列索引 `idx ∈ [0, S)` 精確解碼為 3D 時空座標 `(t, h, w)` 或音頻/文本條件索引
    #[inline]
    pub fn decode_token(&self, idx: usize) -> Result<TokenCoord3D> {
        let total = self.total_tokens();
        if idx >= total {
            bail!("token index {} out of bounds (total S = {})", idx, total);
        }

        let hw = self.h * self.w;
        match self.layout {
            CondTokenLayout::Suffix => {
                let v_len = self.video_tokens();
                if idx < v_len {
                    let t = idx / hw;
                    let rem = idx % hw;
                    let h = rem / self.w;
                    let w = rem % self.w;
                    Ok(TokenCoord3D::Video {
                        t: t as u16,
                        h: h as u16,
                        w: w as u16,
                    })
                } else {
                    Ok(TokenCoord3D::Cond {
                        cond_idx: (idx - v_len) as u32,
                    })
                }
            }
            CondTokenLayout::Prefix => {
                if idx < self.cond_tokens {
                    Ok(TokenCoord3D::Cond {
                        cond_idx: idx as u32,
                    })
                } else {
                    let v_idx = idx - self.cond_tokens;
                    let t = v_idx / hw;
                    let rem = v_idx % hw;
                    let h = rem / self.w;
                    let w = rem % self.w;
                    Ok(TokenCoord3D::Video {
                        t: t as u16,
                        h: h as u16,
                        w: w as u16,
                    })
                }
            }
        }
    }

    /// 將 3D 視頻座標 `(t, h, w)` 編碼回一維序列索引
    #[inline]
    pub fn encode_video(&self, t: usize, h: usize, w: usize) -> Result<usize> {
        if t >= self.t || h >= self.h || w >= self.w {
            bail!(
                "video coord ({}, {}, {}) out of grid ({}, {}, {})",
                t,
                h,
                w,
                self.t,
                self.h,
                self.w
            );
        }
        let v_idx = (t * self.h + h) * self.w + w;
        match self.layout {
            CondTokenLayout::Suffix => Ok(v_idx),
            CondTokenLayout::Prefix => Ok(self.cond_tokens + v_idx),
        }
    }
}

/// 每個 Attention Tile Block (例如 128 Tokens) 的 3D 包圍盒與語義標籤
#[derive(Debug, Clone, Copy)]
pub struct BlockSpan3D {
    pub block_idx: u32,
    pub token_start: usize,
    pub token_end: usize,
    pub has_video: bool,
    pub has_frame0: bool,
    pub has_cond: bool,
    pub t_min: u16,
    pub t_max: u16,
    pub h_min: u16,
    pub h_max: u16,
    pub w_min: u16,
    pub w_max: u16,
}

impl BlockSpan3D {
    /// 兩塊之間在某一維度包圍盒的最小距離（重疊則為 0）
    #[inline]
    fn axis_min_dist(a_min: u16, a_max: u16, b_min: u16, b_max: u16) -> u16 {
        if a_max < b_min {
            b_min - a_max
        } else if b_max < a_min {
            a_min - b_max
        } else {
            0
        }
    }
}

/// 3D 塊稀疏靜態帶狀指數衰減超參數配置
#[derive(Debug, Clone, Copy)]
pub struct RadialDecayConfig {
    /// Block Tile 大小（對齊 FlashAttention2 / SageAttention 的 Br=Bc=128 或 64）
    pub block_size: usize,
    /// 指數衰減係數 λ：R(dt) = max(min_ratio, base_ratio * exp(-lambda * dt))
    pub decay_lambda: f32,
    /// dt = 1 時的初始空間半徑比例（相對於 H 與 W）
    pub base_spatial_ratio: f32,
    /// 遠距離幀保留的最小空間帶寬比例（防止長距離語義完全斷裂）
    pub min_spatial_ratio: f32,
    /// 同幀內（dt == 0）是否強制 100% 稠密空間注意力
    pub intra_frame_dense: bool,
    /// 第 0 幀（t = 0）是否啟用全局雙向錨點（100% Dense）
    pub frame0_global_anchor: bool,
    /// 音頻 / 文本條件 Token 是否啟用 100% Dense 遮罩
    pub cond_tokens_dense: bool,
}

impl Default for RadialDecayConfig {
    fn default() -> Self {
        Self {
            block_size: 128,
            decay_lambda: 0.32,
            base_spatial_ratio: 0.55,
            min_spatial_ratio: 0.125,
            intra_frame_dense: true,
            frame0_global_anchor: true,
            cond_tokens_dense: true,
        }
    }
}

/// 稀疏度與算力節省統計摘要
#[derive(Debug, Clone, Copy)]
pub struct RadialMaskStats {
    pub num_blocks: usize,
    pub total_block_pairs: usize,
    pub active_block_pairs: usize,
    pub cond_dense_pairs: usize,
    pub frame0_anchor_pairs: usize,
    pub radial_band_pairs: usize,
    pub density_ratio: f64,
    pub sparsity_ratio: f64,
    pub theoretical_attn_speedup: f64,
}

/// GPU 側可直接傳入 Block-Sparse Attention 核函數的 CSR 遮罩張量包
pub struct CudaRadialSparseMask {
    pub block_size: usize,
    pub num_blocks: usize,
    /// `[num_blocks + 1]` U32：每個 Query Block 的有效 KV Block 起始偏移
    pub csr_row_ptr: Tensor,
    /// `[active_block_pairs]` U32：壓縮後的活躍 KV Block 索引表
    pub csr_col_idx: Tensor,
    /// `[num_blocks, num_blocks]` U8：稠密二值塊掩碼（1 = 計算，0 = 跳過）
    pub block_mask_u8: Tensor,
}

/// 3D 塊稀疏注意力主結構：封裝時空塊索引、CSR 稀疏表與統計資訊
pub struct RadialAttentionMask {
    pub grid: VideoGrid3D,
    pub config: RadialDecayConfig,
    pub block_spans: Vec<BlockSpan3D>,
    /// 扁平 `[num_blocks, num_blocks]` 塊級遮罩（1 = Active, 0 = Pruned）
    pub block_mask: Vec<u8>,
    /// Block-CSR 行指標（長度 = num_blocks + 1）
    pub csr_row_ptr: Vec<u32>,
    /// Block-CSR 活躍列索引（長度 = stats.active_block_pairs）
    pub csr_col_idx: Vec<u32>,
    /// 第 0 幀全局錨點覆蓋的 Block 索引清單
    pub frame0_anchor_blocks: Vec<u32>,
    /// 音頻/文本 100% Dense 條件覆蓋的 Block 索引清單
    pub cond_dense_blocks: Vec<u32>,
    pub stats: RadialMaskStats,
}

impl RadialAttentionMask {
    /// 構建 3D 塊稀疏靜態掩碼（S=121,041, B=128 時約 946×946 塊，CPU 生成耗時 < 3ms）
    pub fn build(grid: VideoGrid3D, config: RadialDecayConfig) -> Result<Self> {
        if config.block_size == 0 {
            bail!("RadialDecayConfig.block_size must be > 0");
        }
        let total_tokens = grid.total_tokens();
        if total_tokens == 0 {
            bail!("VideoGrid3D total_tokens must be > 0");
        }

        let num_blocks = total_tokens.div_ceil(config.block_size);
        let mut block_spans = Vec::with_capacity(num_blocks);
        let mut frame0_anchor_blocks = Vec::new();
        let mut cond_dense_blocks = Vec::new();

        // 步驟 1：掃描每個 Block 內的所有 Token，建立精確且保守的 3D 包圍盒與語義標記
        for b_idx in 0..num_blocks {
            let token_start = b_idx * config.block_size;
            let token_end = (token_start + config.block_size).min(total_tokens);

            let mut has_video = false;
            let mut has_frame0 = false;
            let mut has_cond = false;
            let mut t_min = u16::MAX;
            let mut t_max = 0u16;
            let mut h_min = u16::MAX;
            let mut h_max = 0u16;
            let mut w_min = u16::MAX;
            let mut w_max = 0u16;

            for tok in token_start..token_end {
                match grid.decode_token(tok)? {
                    TokenCoord3D::Video { t, h, w } => {
                        has_video = true;
                        if t == 0 {
                            has_frame0 = true;
                        }
                        t_min = t_min.min(t);
                        t_max = t_max.max(t);
                        h_min = h_min.min(h);
                        h_max = h_max.max(h);
                        w_min = w_min.min(w);
                        w_max = w_max.max(w);
                    }
                    TokenCoord3D::Cond { .. } => {
                        has_cond = true;
                    }
                }
            }

            if has_frame0 {
                frame0_anchor_blocks.push(b_idx as u32);
            }
            if has_cond {
                cond_dense_blocks.push(b_idx as u32);
            }

            block_spans.push(BlockSpan3D {
                block_idx: b_idx as u32,
                token_start,
                token_end,
                has_video,
                has_frame0,
                has_cond,
                t_min,
                t_max,
                h_min,
                h_max,
                w_min,
                w_max,
            });
        }

        // 步驟 2：預計算時間差 Δt ∈ [0..T) 對應的空間帶狀指數衰減半徑表 (R_h, R_w)
        let mut radius_h_lut = vec![0.0f32; grid.t.max(1)];
        let mut radius_w_lut = vec![0.0f32; grid.t.max(1)];
        for dt in 0..grid.t {
            let ratio = if dt == 0 {
                1.0f32
            } else {
                let decayed = config.base_spatial_ratio * (-config.decay_lambda * dt as f32).exp();
                decayed.max(config.min_spatial_ratio)
            };
            radius_h_lut[dt] = (grid.h as f32 * ratio).max(1.0);
            radius_w_lut[dt] = (grid.w as f32 * ratio).max(1.0);
        }

        // 步驟 3：生成對稱 3D 塊稀疏掩碼與 Block-CSR 索引表
        let total_block_pairs = num_blocks * num_blocks;
        let mut block_mask = vec![0u8; total_block_pairs];
        let mut csr_row_ptr = Vec::with_capacity(num_blocks + 1);
        let mut csr_col_idx = Vec::with_capacity(total_block_pairs / 3);

        let mut active_block_pairs = 0usize;
        let mut cond_dense_pairs = 0usize;
        let mut frame0_anchor_pairs = 0usize;
        let mut radial_band_pairs = 0usize;

        csr_row_ptr.push(0u32);

        for q in 0..num_blocks {
            let bq = &block_spans[q];
            let row_offset = q * num_blocks;

            for k in 0..num_blocks {
                let bk = &block_spans[k];

                // 規則 A：音頻/文本 Token 100% Dense（雙向可見）
                if config.cond_tokens_dense && (bq.has_cond || bk.has_cond) {
                    block_mask[row_offset + k] = 1;
                    csr_col_idx.push(k as u32);
                    active_block_pairs += 1;
                    cond_dense_pairs += 1;
                    continue;
                }

                // 規則 B：第 0 幀全局錨點（雙向可見，防止長序列漂移）
                if config.frame0_global_anchor && (bq.has_frame0 || bk.has_frame0) {
                    block_mask[row_offset + k] = 1;
                    csr_col_idx.push(k as u32);
                    active_block_pairs += 1;
                    frame0_anchor_pairs += 1;
                    continue;
                }

                // 規則 C：靜態帶狀指數衰減（Radial Spatiotemporal Cone）
                if bq.has_video && bk.has_video {
                    let dt = BlockSpan3D::axis_min_dist(bq.t_min, bq.t_max, bk.t_min, bk.t_max)
                        as usize;

                    let is_active = if dt == 0 && config.intra_frame_dense {
                        true
                    } else {
                        let dh =
                            BlockSpan3D::axis_min_dist(bq.h_min, bq.h_max, bk.h_min, bk.h_max)
                                as f32;
                        let dw =
                            BlockSpan3D::axis_min_dist(bq.w_min, bq.w_max, bk.w_min, bk.w_max)
                                as f32;
                        let rh = radius_h_lut[dt.min(radius_h_lut.len() - 1)];
                        let rw = radius_w_lut[dt.min(radius_w_lut.len() - 1)];
                        let norm_h = dh / rh;
                        let norm_w = dw / rw;
                        (norm_h * norm_h + norm_w * norm_w) <= 1.0
                    };

                    if is_active {
                        block_mask[row_offset + k] = 1;
                        csr_col_idx.push(k as u32);
                        active_block_pairs += 1;
                        radial_band_pairs += 1;
                    }
                }
            }

            csr_row_ptr.push(csr_col_idx.len() as u32);
        }

        let density_ratio = active_block_pairs as f64 / total_block_pairs as f64;
        let sparsity_ratio = 1.0 - density_ratio;
        let theoretical_attn_speedup = 1.0 / density_ratio.max(1e-9);

        let stats = RadialMaskStats {
            num_blocks,
            total_block_pairs,
            active_block_pairs,
            cond_dense_pairs,
            frame0_anchor_pairs,
            radial_band_pairs,
            density_ratio,
            sparsity_ratio,
            theoretical_attn_speedup,
        };

        Ok(Self {
            grid,
            config,
            block_spans,
            block_mask,
            csr_row_ptr,
            csr_col_idx,
            frame0_anchor_blocks,
            cond_dense_blocks,
            stats,
        })
    }

    /// 查詢指定 (q_block, k_block) 是否為活躍計算塊
    #[inline]
    pub fn is_block_active(&self, q_block: usize, k_block: usize) -> bool {
        self.block_mask[q_block * self.stats.num_blocks + k_block] != 0
    }

    /// 獲取指定 Query Block 對應的所有活躍 KV Block 切片
    #[inline]
    pub fn active_kv_blocks(&self, q_block: usize) -> &[u32] {
        let start = self.csr_row_ptr[q_block] as usize;
        let end = self.csr_row_ptr[q_block + 1] as usize;
        &self.csr_col_idx[start..end]
    }

    /// 將 CSR 索引與塊掩碼上載至 CUDA 顯存（1088P 8s 總顯存佔用 < 2.5 MB）
    pub fn to_cuda_mask(&self, device: &Device) -> Result<CudaRadialSparseMask> {
        let num_blocks = self.stats.num_blocks;
        let csr_row_ptr = Tensor::from_slice(&self.csr_row_ptr, num_blocks + 1, device)?;
        let csr_col_idx = Tensor::from_slice(&self.csr_col_idx, self.csr_col_idx.len(), device)?;
        let block_mask_u8 =
            Tensor::from_slice(&self.block_mask, (num_blocks, num_blocks), device)?;

        Ok(CudaRadialSparseMask {
            block_size: self.config.block_size,
            num_blocks,
            csr_row_ptr,
            csr_col_idx,
            block_mask_u8,
        })
    }
}

#[cfg(test)]
mod radial_mask_tests {
    use super::*;

    #[test]
    fn test_h3_1088p_8s_radial_mask_invariants() -> Result<()> {
        let grid = VideoGrid3D::h3_1088p_8s(CondTokenLayout::Suffix);
        assert_eq!(grid.video_tokens(), 120_700);
        assert_eq!(grid.total_tokens(), 121_041);

        // 驗證邊界座標精確解碼
        assert_eq!(
            grid.decode_token(0)?,
            TokenCoord3D::Video { t: 0, h: 0, w: 0 }
        );
        assert_eq!(
            grid.decode_token(120_699)?,
            TokenCoord3D::Video {
                t: 24,
                h: 67,
                w: 70
            }
        );
        assert_eq!(
            grid.decode_token(120_700)?,
            TokenCoord3D::Cond { cond_idx: 0 }
        );
        assert_eq!(
            grid.decode_token(121_040)?,
            TokenCoord3D::Cond { cond_idx: 340 }
        );

        let mask = RadialAttentionMask::build(grid, RadialDecayConfig::default())?;
        assert_eq!(mask.stats.num_blocks, 946);

        // 驗證第 0 幀全局錨點 100% 稠密
        for &f0_blk in &mask.frame0_anchor_blocks {
            assert_eq!(
                mask.active_kv_blocks(f0_blk as usize).len(),
                mask.stats.num_blocks
            );
        }

        // 驗證音頻/文本 Token 100% 稠密
        for &cond_blk in &mask.cond_dense_blocks {
            assert_eq!(
                mask.active_kv_blocks(cond_blk as usize).len(),
                mask.stats.num_blocks
            );
        }

        // 驗證對稱性與有效稀疏壓縮
        for q in 0..mask.stats.num_blocks {
            for k in 0..mask.stats.num_blocks {
                assert_eq!(mask.is_block_active(q, k), mask.is_block_active(k, q));
            }
        }
        assert!(
            mask.stats.sparsity_ratio > 0.40,
            "expected significant 3D block sparsity, got {:.2}%",
            mask.stats.sparsity_ratio * 100.0
        );
        Ok(())
    }
}
