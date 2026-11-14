//! Memory configuration: presets for different VRAM / RAM budgets, loadable from TOML.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::weights::Placement;

/// 3-Tier weight allocation distribution across VRAM, pinned host RAM, and NVMe disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TierDistribution {
    pub vram: usize,
    pub ram: usize,
    pub disk: usize,
}

impl TierDistribution {
    pub fn new(vram: usize, ram: usize, disk: usize) -> Self {
        Self { vram, ram, disk }
    }
}

/// How the pipeline spends VRAM and RAM. Every field has a preset default; see `configs/*.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MemoryConfig {
    /// DiT transformer blocks (of 50, ~0.39 GB each as int8) kept resident on the GPU. The rest
    /// are uploaded from `offload` every sampling step. `-1` = as many as fit (auto).
    pub dit_gpu_blocks: i64,
    /// Explicit count of DiT blocks placed in Tier 1 (VRAM). Default -1 = auto-resolve.
    pub dit_vram_blocks: i64,
    /// Explicit count of DiT blocks placed in Tier 2 (pinned host RAM). Default -1 = auto-resolve.
    pub dit_ram_blocks: i64,
    /// Explicit count of DiT blocks placed in Tier 3 (NVMe disk). Default -1 = remainder of blocks.
    pub dit_disk_blocks: i64,
    /// 3-Tier allocation ratio string (e.g. "23:14:10" or "0.5:0.3:0.2"). If set, overrides explicit counts.
    pub tier_ratio: Option<String>,
    /// Qwen3-VL text encoder layers (of 50, ~0.31 GB each as nvfp4) resident while encoding.
    /// `-1` = auto.
    pub te_gpu_layers: i64,
    /// Where non-resident weights wait between uses: `ram` (a copy in process memory, fastest
    /// upload) or `disk` (mmapped file / page cache, lowest RAM use, slowest when RAM is short).
    pub offload: Placement,
    /// Placement of the text encoder's non-resident layers (default: same as `offload`). It runs once per
    /// request, so `disk` keeps it out of pinned RAM at little cost.
    pub te_offload: Option<Placement>,
    /// Free each model as soon as its stage is done (text encoder -> DiT -> VAEs), so only one
    /// stage's weights are resident at a time.
    pub sequential: bool,
    /// Max token rows per GEMM in the DiT (bounds activation memory of the big projections).
    pub token_chunk: usize,
    /// Max token rows per MLP chunk in the DiT (the fc1 output is 5.3x wider than the stream).
    pub mlp_chunk: usize,
    /// VRAM kept free for activations when `-1` (auto) budgets are resolved, in GB.
    pub reserve_gb: f64,
    /// Video VAE tiles (256 px) computed together per decoder layer.
    pub vae_tile_batch: usize,
    /// Keep the video VAE decoder (~4.8 GB fp16) resident; otherwise its layers stream from
    /// `offload` once per 17-frame chunk.
    pub vae_resident: bool,
    /// Compute dtype of the video VAE: `f16` (ComfyUI default), `bf16` or `f32`.
    pub vae_dtype: String,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Preset::Gb24.config()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Preset {
    /// Auto-fit to the current free VRAM, weights offloaded to RAM.
    Auto,
    /// 24 GB GPU (4090 / 3090) with >= 48 GB RAM.
    Gb24,
    /// 16 GB GPU with >= 32 GB RAM.
    Gb16,
    /// 12 GB GPU with >= 32 GB RAM.
    Gb12,
    /// 8 GB GPU; weights stream from disk, low RAM (>= 16 GB) use.
    Gb8,
    /// Smallest footprint: nothing resident, everything streamed from disk, small chunks.
    Minimal,
}

impl Preset {
    pub fn config(self) -> MemoryConfig {
        let base = MemoryConfig {
            dit_gpu_blocks: -1,
            dit_vram_blocks: -1,
            dit_ram_blocks: -1,
            dit_disk_blocks: -1,
            tier_ratio: None,
            te_gpu_layers: 50,
            offload: Placement::Ram,
            te_offload: None,
            sequential: true,
            token_chunk: 16384,
            mlp_chunk: 8192,
            reserve_gb: 6.0,
            vae_tile_batch: 4,
            vae_resident: true,
            vae_dtype: "f16".into(),
        };
        match self {
            Preset::Auto => MemoryConfig { dit_gpu_blocks: -1, te_gpu_layers: -1, ..base },
            Preset::Gb24 => base,
            Preset::Gb16 => MemoryConfig { dit_gpu_blocks: 14, te_gpu_layers: 30, token_chunk: 8192, mlp_chunk: 4096, vae_tile_batch: 2, ..base },
            Preset::Gb12 => MemoryConfig { dit_gpu_blocks: 4, te_gpu_layers: 16, token_chunk: 8192, mlp_chunk: 2048, vae_tile_batch: 1, ..base },
            Preset::Gb8 => MemoryConfig {
                dit_gpu_blocks: 0,
                te_gpu_layers: 4,
                offload: Placement::Disk,
                token_chunk: 4096,
                mlp_chunk: 1024,
                vae_tile_batch: 1,
                ..base
            },
            Preset::Minimal => MemoryConfig {
                dit_gpu_blocks: 0,
                te_gpu_layers: 0,
                offload: Placement::Disk,
                token_chunk: 2048,
                mlp_chunk: 512,
                vae_tile_batch: 1,
                vae_resident: false,
                ..base
            },
        }
    }
}

impl MemoryConfig {
    pub fn from_toml(path: &str) -> Result<Self> {
        let s = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
        #[derive(Deserialize)]
        struct File {
            memory: Option<MemoryConfig>,
        }
        let f: File = toml::from_str(&s).with_context(|| format!("parsing {path}"))?;
        f.memory.context("config file needs a [memory] table")
    }

    pub fn vae_dtype(&self) -> Result<candle_core::DType> {
        Ok(match self.vae_dtype.as_str() {
            "f16" | "fp16" => candle_core::DType::F16,
            "bf16" => candle_core::DType::BF16,
            "f32" | "fp32" => candle_core::DType::F32,
            d => bail!("vae_dtype must be f16, bf16 or f32, got {d}"),
        })
    }
}

/// Free / total VRAM in bytes on the current CUDA context.
pub fn vram_info() -> Result<(usize, usize)> {
    crate::kernels::trim_pool();
    #[cfg(feature = "cuda")]
    { return candle_core::cuda_backend::cudarc::driver::result::mem_get_info().map_err(|e| anyhow::anyhow!("mem_get_info: {e:?}")); }
    #[cfg(feature = "mlx")]
    { return crate::apple::memory_info(); }
    #[cfg(not(any(feature = "cuda", feature = "mlx")))]
    { bail!("select the cuda or mlx backend"); }
}

/// Available system host RAM in bytes.
pub fn available_ram() -> usize {
    if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
        for line in content.lines() {
            if line.starts_with("MemAvailable:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(kb) = parts[1].parse::<usize>() {
                        return kb * 1024;
                    }
                }
            }
        }
    }
    4 * 1024 * 1024 * 1024
}

/// Resolve 3-tier distribution (VRAM, RAM, Disk) across total layers.
pub fn resolve_tiers(
    total: usize,
    vram_req: i64,
    ram_req: i64,
    disk_req: i64,
    tier_ratio: Option<&str>,
    unit_bytes: usize,
    vram_reserve_bytes: usize,
    ram_reserve_bytes: usize,
) -> TierDistribution {
    if let Some(r) = tier_ratio {
        let parts: Vec<f64> = r.split(':').filter_map(|s| s.parse().ok()).collect();
        if parts.len() == 3 {
            let sum: f64 = parts.iter().sum();
            if sum > 0.0 {
                let v = ((parts[0] / sum) * total as f64).round() as usize;
                let v = v.min(total);
                let r_alloc = ((parts[1] / sum) * total as f64).round() as usize;
                let r_alloc = r_alloc.min(total - v);
                let d = total - v - r_alloc;
                return TierDistribution::new(v, r_alloc, d);
            }
        }
    }

    // 1. Resolve VRAM
    let vram = if vram_req >= 0 {
        (vram_req as usize).min(total)
    } else {
        let free_vram = vram_info().map(|(f, _)| f).unwrap_or(0);
        (free_vram.saturating_sub(vram_reserve_bytes) / unit_bytes.max(1)).min(total)
    };

    let remaining = total - vram;

    // 2. Resolve RAM
    let ram = if ram_req >= 0 {
        (ram_req as usize).min(remaining)
    } else {
        let avail_ram = available_ram();
        let safe_ram = avail_ram.saturating_sub(ram_reserve_bytes);
        (safe_ram / unit_bytes.max(1)).min(remaining)
    };

    // 3. Resolve Disk
    let disk = if disk_req >= 0 {
        (disk_req as usize).min(total - vram - ram)
    } else {
        total - vram - ram
    };

    TierDistribution::new(vram, ram, disk)
}

/// Resolve a "count of resident units" budget: explicit values clamp to `total`, `-1` fits as many
/// `unit_bytes` units as free VRAM allows after `reserve` bytes.
pub fn resolve_resident(requested: i64, total: usize, unit_bytes: usize, reserve: usize) -> usize {
    if requested >= 0 {
        return (requested as usize).min(total);
    }
    let free = vram_info().map(|(f, _)| f).unwrap_or(0);
    (free.saturating_sub(reserve) / unit_bytes.max(1)).min(total)
}
