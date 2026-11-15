//! MiniMax H3 audio-video DiT (port of `comfy/ldm/minimax/model.py`).
//!
//! Single-stream packed transformer over `[text | cond rows | audio | video]`, conditioned on
//! Qwen3-VL hidden states, denoising 24ch video latents (patch 1x2x2) and 32ch stereo audio
//! latents (40 Hz) jointly. Video and audio run on different flow shifts (12 / 3); the sampler
//! carries the audio scaled onto the video schedule and [`Dit::forward`] converts in and out.

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, D};

use crate::config::MemoryConfig;
use crate::kernels;
use crate::lora::Lora;
use crate::ops;
use crate::weights::{Held, Placement, QFormat, QLinear, SafeFile};

pub const FRAME_PER_TOKEN: [usize; 5] = [1, 4, 4, 4, 4];
pub const FRAME_RESCALE: f64 = 5.0 / 3.0;
pub const VISUAL_COND_TIMESTEP: f64 = 0.999;
pub const AUDIO_COND_TIMESTEP: f64 = 1.0;

pub fn time_shift_sigma(sigma: f64, from_shift: f64, to_shift: f64) -> f64 {
    let base = sigma / (from_shift + sigma * (1.0 - from_shift));
    to_shift * base / (1.0 + (to_shift - 1.0) * base)
}

// ----------------------------------------------------------------------------------------------
// Packed layout
// ----------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seg {
    Text,
    Cond,
    CondAudio,
    RefImg,
    RefAudio,
    Audio,
    Video,
}

impl Seg {
    fn tag(self) -> u32 {
        match self {
            Seg::Text => 1,
            Seg::Video | Seg::Cond | Seg::RefImg => 0,
            Seg::Audio | Seg::CondAudio | Seg::RefAudio => 2,
        }
    }
    fn is_video(self) -> bool {
        matches!(self, Seg::Cond | Seg::RefImg | Seg::Video)
    }
}

/// Keyframe condition shape info (fl2va / AddGuide).
#[derive(Debug, Clone)]
pub struct KeyframeSpec {
    pub frame_index: usize,
    pub video_t: Option<usize>,
    pub audio_t: Option<usize>,
}

/// Reference block shape info (ref2va).
#[derive(Debug, Clone)]
pub enum RefSpec {
    Image { lh: usize, lw: usize },
    Audio { rt: usize },
    Video { lt: usize, lh: usize, lw: usize, rt: usize },
}

impl RefSpec {
    fn t_span(&self) -> f64 {
        match self {
            RefSpec::Image { .. } => 1.0,
            RefSpec::Audio { rt } => *rt as f64,
            RefSpec::Video { lt, rt, .. } => (*rt as f64).max(video_t_spans(*lt).iter().sum()),
        }
    }
}

fn axis_from_sqrt_area(dim: usize, patch: usize, sqrt_area: f64) -> Vec<f64> {
    let ratio = dim as f64 / sqrt_area;
    let n = dim / patch;
    (0..n).map(|k| (k as f64 * (ratio / n as f64) + (1.0 - ratio) / 2.0) * 32.0).collect()
}

/// Area-normalized (h, w) coordinates of one latent frame's 2x2-patch rows + the w axis.
fn frame_grid(h: usize, w: usize) -> (Vec<[f64; 2]>, Vec<f64>) {
    let area = ((h * w) as f64).sqrt();
    let hh = axis_from_sqrt_area(h, 2, area);
    let ww = axis_from_sqrt_area(w, 2, area);
    let mut rows = Vec::with_capacity(hh.len() * ww.len());
    for &a in &hh {
        for &b in &ww {
            rows.push([a, b]);
        }
    }
    (rows, ww)
}

fn video_t_spans(n: usize) -> Vec<f64> {
    (0..n).map(|k| FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] as f64).collect()
}

fn video_t_grid(n: usize, origin: f64) -> Vec<f64> {
    let spans = video_t_spans(n);
    let mut out = Vec::with_capacity(n);
    let mut acc = 0.0;
    for s in spans {
        out.push(origin + acc);
        acc += s;
    }
    out
}

fn audio_grid(cursor: f64, t: usize, w_low: f64, w_high: f64) -> Vec<[f64; 3]> {
    let mut g = Vec::with_capacity(2 * t);
    for ch in 0..2 {
        for k in 0..t {
            g.push([cursor + k as f64, 0.0, if ch == 0 { w_low } else { w_high }]);
        }
    }
    g
}

fn video_grid(vt: usize, frame: &[[f64; 2]], cursor: f64) -> Vec<[f64; 3]> {
    let tg = video_t_grid(vt, cursor);
    let mut g = Vec::with_capacity(vt * frame.len());
    for t in tg {
        for f in frame {
            g.push([t, f[0], f[1]]);
        }
    }
    g
}

#[derive(Debug, Clone)]
pub struct PackedLayout {
    pub seq_len: usize,
    pub positions: Vec<[f64; 3]>,
    /// Contiguous `(start, stop, kind)` segments covering the sequence.
    pub segments: Vec<(usize, usize, Seg)>,
}

impl PackedLayout {
    pub fn new(
        text_len: usize,
        latent_t: usize,
        latent_h: usize,
        latent_w: usize,
        audio_t: usize,
        keyframes: &[KeyframeSpec],
        refs: &[RefSpec],
    ) -> Self {
        let (frame, w_grid) = frame_grid(latent_h, latent_w);
        let mut segs: Vec<(Seg, usize)> = vec![(Seg::Text, text_len)];
        let mut pos: Vec<[f64; 3]> = (0..text_len).map(|k| [k as f64, 0.0, 0.0]).collect();
        let target_w = (w_grid[0], *w_grid.last().unwrap());
        let mut cursor = text_len as f64 + refs.iter().map(|r| r.t_span()).sum::<f64>();

        for kf in keyframes {
            let cond_t = cursor + FRAME_RESCALE * kf.frame_index as f64;
            if let Some(vt) = kf.video_t {
                segs.push((Seg::Cond, vt * frame.len()));
                pos.extend(video_grid(vt, &frame, cond_t));
            }
            if let Some(rt) = kf.audio_t {
                segs.push((Seg::CondAudio, rt * 2));
                pos.extend(audio_grid(cond_t, rt, target_w.0, target_w.1));
            }
        }

        if !refs.is_empty() {
            let mut c = text_len as f64;
            for r in refs {
                match *r {
                    RefSpec::Image { lh, lw } => {
                        let (rf, _) = frame_grid(lh, lw);
                        segs.push((Seg::RefImg, rf.len()));
                        pos.extend(rf.iter().map(|f| [c, f[0], f[1]]));
                        c += 1.0;
                    }
                    RefSpec::Audio { rt } => {
                        if rt > 0 {
                            segs.push((Seg::RefAudio, rt * 2));
                            pos.extend(audio_grid(c, rt, target_w.0, target_w.1));
                        }
                        c += rt as f64;
                    }
                    RefSpec::Video { lt, lh, lw, rt } => {
                        let (rf, rw) = frame_grid(lh, lw);
                        if rt > 0 {
                            segs.push((Seg::RefAudio, rt * 2));
                            pos.extend(audio_grid(c, rt, rw[0], *rw.last().unwrap()));
                        }
                        segs.push((Seg::RefImg, lt * rf.len()));
                        pos.extend(video_grid(lt, &rf, c));
                        c += (rt as f64).max(video_t_spans(lt).iter().sum());
                    }
                }
            }
            cursor = c;
        }

        segs.push((Seg::Audio, audio_t * 2));
        pos.extend(audio_grid(cursor, audio_t, target_w.0, target_w.1));
        segs.push((Seg::Video, latent_t * frame.len()));
        pos.extend(video_grid(latent_t, &frame, cursor));

        let mut segments = vec![];
        let mut off = 0;
        for (k, n) in segs {
            segments.push((off, off + n, k));
            off += n;
        }
        Self { seq_len: off, positions: pos, segments }
    }

    pub fn find(&self, kind: Seg) -> (usize, usize) {
        let s = self.segments.iter().find(|s| s.2 == kind).unwrap();
        (s.0, s.1)
    }
}

// ----------------------------------------------------------------------------------------------
// Patchify helpers
// ----------------------------------------------------------------------------------------------

/// `[1, C, T, H, W]` -> `[T*h*w, C*2*2]` (patch 1x2x2).
pub fn patchify_video(x: &Tensor) -> Result<Tensor> {
    let (b, c, t, h, w) = x.dims5()?;
    let x = x.reshape(&[b, c, t, 1, h / 2, 2, w / 2, 2][..])?;
    let x = x.permute(vec![0, 2, 4, 6, 1, 3, 5, 7])?.contiguous()?;
    Ok(x.reshape((b * t * (h / 2) * (w / 2), c * 4))?)
}

pub fn unpatchify_video(rows: &Tensor, t: usize, h: usize, w: usize, c: usize) -> Result<Tensor> {
    let x = rows.reshape(&[1, t, h, w, c, 1, 2, 2][..])?;
    let x = x.permute(vec![0, 4, 1, 5, 2, 6, 3, 7])?.contiguous()?;
    Ok(x.reshape((1, c, t, h * 2, w * 2))?)
}

/// `[1, C, 2, T]` -> `[2T, C]` channel-major.
pub fn pack_audio(x: &Tensor) -> Result<Tensor> {
    let (_, c, ch, t) = x.dims4()?;
    Ok(x.get(0)?.permute((1, 2, 0))?.contiguous()?.reshape((ch * t, c))?)
}

pub fn unpack_audio(rows: &Tensor) -> Result<Tensor> {
    let (n, c) = rows.dims2()?;
    Ok(rows.reshape((2, n / 2, c))?.permute((2, 0, 1))?.contiguous()?.unsqueeze(0)?)
}

// ----------------------------------------------------------------------------------------------
// Model
// ----------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DitConfig {
    pub hidden: usize,
    pub layers: usize,
    pub refiner_layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub latents_dim: usize,
    /// Stacked PDD output heads in the checkpoint itself (1 = plain heads; a LoRA can grow them later).
    pub pdd_heads: usize,
    pub audio_dim: usize,
    pub text_dim: usize,
    pub time_embed_dim: usize,
    pub curve_grid: Option<usize>,
    pub timestep_input_dim: usize,
    pub eps: f32,
    pub shift_video: f64,
    pub shift_audio: f64,
}

impl DitConfig {
    /// Same detection as `comfy/model_detection.py` for MiniMax H3.
    pub fn detect(f: &SafeFile) -> Result<Self> {
        if !f.has("video_patch_proj.weight") || !f.has("audio_patch_proj.weight") {
            bail!("{} is not a MiniMax H3 diffusion model", f.path);
        }
        let count = |p: &str, suffix: &str| (0..).take_while(|i| f.has(&format!("{p}{i}.{suffix}"))).count();
        let hidden = f.shape("video_patch_proj.weight")?[0];
        let head_dim = f.shape("blocks.0.attn.q_norm.weight")?[0];
        let qkv_shape = f.shape("blocks.0.attn.qkv_proj.weight")?;
        let qkv = if qkv_shape.len() == 4 { qkv_shape[0] * 8 } else { qkv_shape[0] };
        let fc1_shape = f.shape("blocks.0.mlp.fc1.weight")?;
        let fc1_dim0 = if fc1_shape.len() == 4 { fc1_shape[0] * 8 } else { fc1_shape[0] };
        let (time_embed_dim, curve_grid, tin) = if f.has("adaln_t_table") {
            let s = f.shape("adaln_t_table")?;
            (s[1], Some(s[0]), 256)
        } else {
            (f.shape("time_embedder.proj_out.weight")?[0], None, f.shape("time_embedder.proj_in.weight")?[1])
        };
        // a native PDD checkpoint ships `pdd_steps` stacked output heads (block 0 full, the rest offsets)
        let pdd_heads = f
            .metadata
            .get("config")
            .and_then(|c| serde_json::from_str::<serde_json::Value>(c).ok())
            .and_then(|c| c["transformer"]["pdd_steps"].as_u64())
            .map_or(1, |n| (n as usize).max(1));
        Ok(Self {
            hidden,
            pdd_heads,
            layers: count("blocks.", "norm1.weight"),
            refiner_layers: count("token_refiner.blocks.", "norm1.weight"),
            heads: qkv / (3 * head_dim),
            head_dim,
            ffn: fc1_dim0 / 2,
            latents_dim: f.shape("final_layer.video_out.weight")?[0] / pdd_heads / 4,
            audio_dim: f.shape("final_layer.audio_out.weight")?[0] / pdd_heads,
            text_dim: f.shape("condition_proj.weight")?[1],
            time_embed_dim,
            curve_grid,
            timestep_input_dim: tin,
            eps: 1e-5,
            shift_video: 12.0,
            shift_audio: 3.0,
        })
    }
}

#[derive(Clone)]
pub(crate) struct LowRankLinear {
    pub(crate) down: QLinear,
    pub(crate) up: QLinear,
}

impl LowRankLinear {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mid = self.down.forward(x)?;
        self.up.forward(&mid)
    }
}

pub(crate) enum QkvProj {
    Fused(QLinear),
    Factorized {
        q: LowRankLinear,
        k: LowRankLinear,
        v: LowRankLinear,
    },
}

impl QkvProj {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            QkvProj::Fused(qkv) => qkv.forward(x),
            QkvProj::Factorized { q, k, v } => {
                let q_out = q.forward(x)?;
                let k_out = k.forward(x)?;
                let v_out = v.forward(x)?;
                Ok(Tensor::cat(&[&q_out, &k_out, &v_out], 1)?)
            }
        }
    }
}

pub(crate) enum LinProj {
    Single(QLinear),
    Factorized(LowRankLinear),
}

impl LinProj {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            LinProj::Single(lin) => lin.forward(x),
            LinProj::Factorized(lr) => lr.forward(x),
        }
    }
}

pub(crate) struct Block {
    pub(crate) norm1: Tensor,
    pub(crate) norm2: Tensor,
    pub(crate) q_norm: Tensor,
    pub(crate) k_norm: Tensor,
    pub(crate) qkv: QkvProj,
    pub(crate) out: LinProj,
    pub(crate) fc1: LinProj,
    pub(crate) fc2: LinProj,
    pub(crate) adaln: QLinear,
}

/// Per-forward state a block needs besides its weights.
pub(crate) struct StepCtx<'a> {
    /// Timestep embeddings of the distinct timesteps, `[U, t_dim]` f32.
    pub t_emb: &'a Tensor,
    /// Curve-form adaLN (no SiLU before the projection).
    pub curve: bool,
    /// Per-token row of the `[3U, hidden]` modulation tables.
    pub idx: &'a Tensor,
    pub rope: &'a Tensor,
    pub u: usize,
}

struct RefinerBlock {
    norm1: Tensor,
    norm2: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    qkv: QLinear,
    out: QLinear,
    fc1: QLinear,
    fc2: QLinear,
}

/// Largest fused qkv (bytes) a block's attention materializes at once; longer sequences go in head groups.
const ATTN_QKV_BUDGET: usize = 1_500_000_000;

pub struct Dit {
    pub cfg: DitConfig,
    dev: Device,
    dtype: DType,
    mem: MemoryConfig,
    video_patch_proj: QLinear,
    audio_patch_proj: QLinear,
    condition_proj: QLinear,
    adaln_t_table: Option<Tensor>,
    time_embedder: Option<(QLinear, QLinear)>,
    inv_freq: Tensor,
    refiner: Vec<RefinerBlock>,
    refiner_norm: Tensor,
    blocks: Vec<Block>,
    /// Blocks whose weights sit in VRAM now, and how many were at load (`set_resident` moves in between).
    resident: usize,
    base_resident: usize,
    pub tiers: crate::config::TierDistribution,
    block_bytes: usize,
    final_norm: Tensor,
    final_adaln: QLinear,
    video_out: QLinear,
    audio_out: QLinear,
    /// PDD head bank size of the output heads (1 = plain heads).
    pdd_heads: usize,
    control: Option<crate::control::Control>,
    /// Debug: block indices whose outputs are recorded into `captured`.
    pub capture_blocks: Vec<usize>,
    pub captured: std::cell::RefCell<Vec<(usize, Tensor)>>,
    #[cfg(feature = "cuda")]
    pub streaming: Option<crate::streaming::StreamingEngine>,
}

/// Conditioning for one sampling run.
#[derive(Clone)]
pub struct DitCond {
    /// Qwen3-VL hidden states `[L, text_dim]` (or already refined `[L, hidden]`).
    pub text: Tensor,
    /// Per-text-token adaLN tag (0 = vision block, 1 = text).
    pub tags: Option<Vec<u32>>,
    pub keyframes: Vec<KeyframeCond>,
    pub refs: Vec<RefCond>,
    pub seed: u64,
    pub visual_cond_noise_aug: f64,
    pub audio_cond_noise_aug: f64,
    /// Denoise mask on the 2x2-patch token grid (see [`TokenMasks`]).
    pub denoise: Option<TokenMasks>,
    /// Fun ControlNet hint and schedule (needs [`Dit::set_control`]).
    pub control: Option<ControlCond>,
}

/// A denoise mask (1 = generate, 0 = keep) pooled to the DiT's token grid, as ComfyUI's
/// `MiniMaxH3._token_grid_masks`: video max-pooled per 2x2 latent patch, every value rounded up to a multiple of
/// 1/256. A stream is `None` when it generates everywhere. Rows with a mask value below 1 run at their own
/// timestep and their velocity is scaled by the value.
#[derive(Clone, Default)]
pub struct TokenMasks {
    /// `[T, H, W]` latent cells, row-major.
    pub video: Option<Vec<f32>>,
    /// One value per audio latent frame.
    pub audio: Option<Vec<f32>>,
}

/// A Fun ControlNet hint for one sampling pass.
#[derive(Clone)]
pub struct ControlCond {
    /// Hint latent `[1, C, T, H, W]` (24 channels, or 49 with the inpaint visibility mask and masked source).
    pub latent: Tensor,
    pub strength: f64,
    /// The ControlNet runs while `sigma_end <= sigma <= sigma_start`.
    pub sigma_start: f64,
    pub sigma_end: f64,
}

#[derive(Clone)]
pub struct KeyframeCond {
    pub frame_index: usize,
    /// Normalized video latent `[1, 24, t, h, w]`.
    pub video: Option<Tensor>,
    /// Normalized audio latent `[1, 32, 2, t]`.
    pub audio: Option<Tensor>,
}

#[derive(Clone)]
pub enum RefCond {
    Image { latent: Tensor },
    Audio { latent: Tensor },
    Video { latent: Tensor, audio: Option<Tensor> },
}

/// Per-run state: packed layout, rope table, refined text and condition rows.
pub struct DitRun {
    layout: PackedLayout,
    rope: Tensor,
    text: Tensor,
    tags: Option<Vec<u32>>,
    cond_video_rows: Option<Tensor>,
    cond_audio_rows: Option<Tensor>,
    vis_aug: f64,
    aud_aug: f64,
    latent_t: usize,
    lat_h: usize,
    lat_w: usize,
    audio_t: usize,
    /// Per-2x2-patch-row mask values of the video stream, and per-audio-row (`2 * audio_t`, channel-major), for
    /// the per-row timesteps; `*_out` scale the predicted velocity (`[1, 1, T, H, W]`, `[1, 1, 2, audio_t]`).
    vmask_rows: Option<Vec<f32>>,
    vmask_out: Option<Tensor>,
    amask_rows: Option<Vec<f32>>,
    amask_out: Option<Tensor>,
    control: Option<RunControl>,
    pub sample_sigmas: Vec<f64>,
}

impl DitRun {
    /// Tokens of the packed sequence (video, audio, text, keyframe and reference rows).
    pub fn seq_len(&self) -> usize {
        self.layout.seq_len
    }
}

pub(crate) struct RunControl {
    pub(crate) cond: ControlCond,
    /// Control patch rows for every video-kind row of the packed sequence (zeros for keyframe / reference rows).
    pub(crate) rows: Tensor,
    /// `[S, 1]`: the strength for rows the control adds to, 0 for audio rows.
    pub(crate) scale: Tensor,
}

fn ld(f: &Arc<SafeFile>, name: &str, dev: &Device, dt: DType) -> Result<Tensor> {
    f.load_as(name, dev, dt)
}

fn dense_f32(f: &Arc<SafeFile>, prefix: &str, dev: &Device) -> Result<QLinear> {
    let mut l = QLinear::load(f, prefix, Placement::Gpu, dev, DType::F32)?;
    if let QFormat::Dense = l.fmt {
        l.weight = Held::from_tensor(l.weight.get(dev)?.to_dtype(DType::F32)?);
    }
    Ok(l)
}

fn cpu_chunk_to_device(t: &Tensor, dev: &Device) -> Result<Tensor> {
    if t.device().is_cuda() {
        return Ok(t.clone());
    }
    let shape = t.shape().clone();
    let flat = t.flatten_all()?;
    match t.dtype() {
        DType::BF16 => {
            let v = flat.to_vec1::<half::bf16>()?;
            Ok(Tensor::from_vec(v, shape, dev)?)
        }
        DType::F16 => {
            let v = flat.to_vec1::<half::f16>()?;
            Ok(Tensor::from_vec(v, shape, dev)?)
        }
        _ => {
            let v = flat.to_vec1::<f32>()?;
            Ok(Tensor::from_vec(v, shape, dev)?)
        }
    }
}


pub(crate) enum AttResult {
    Gpu(Tensor),
    Pinned(std::sync::Arc<crate::pinned_att::PinnedAttBuffer>),
}

impl Dit {
    pub fn load(path: &str, dev: &Device, mem: &MemoryConfig, lora: Option<(&Lora, f64)>) -> Result<Self> {
        let f = SafeFile::open(path)?;
        let cfg = DitConfig::detect(&f)?;
        let dtype = DType::BF16;
        log::info(&format!(
            "DiT: {} blocks, hidden {}, {} heads x {}, adaln {}",
            cfg.layers,
            cfg.hidden,
            cfg.heads,
            cfg.head_dim,
            if cfg.curve_grid.is_some() { "curve" } else { "time embedder" }
        ));
        let block_bytes: usize = ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"]
            .iter()
            .map(|p| f.bytes(&format!("blocks.0.{p}.weight")).map(|b| b.len()).unwrap_or(0))
            .sum();
        let lora_bytes = lora.map(|(l, _)| l.device_bytes()).unwrap_or(0);
        let vram_req = if mem.dit_vram_blocks != -1 { mem.dit_vram_blocks } else { mem.dit_gpu_blocks };
        let tiers = crate::config::resolve_tiers(
            cfg.layers,
            vram_req,
            mem.dit_ram_blocks,
            mem.dit_disk_blocks,
            mem.tier_ratio.as_deref(),
            block_bytes,
            (mem.reserve_gb * 1e9) as usize + lora_bytes + 500_000_000,
            1_000_000_000,
        );
        let resident = tiers.vram;
        #[cfg(feature = "cuda")]
        crate::kernels::keep_pool_memory(dev);
        #[cfg(feature = "cuda")]
        let is_w4a4_preswizzled = f.metadata.get("h3.quant.format").map(|s| s == "ampere_w4a4_group256_preswizzled").unwrap_or(false)
            || f.has("blocks.0.attn.qkv_proj.weight_scale");
        let streaming = if dev.is_cuda() && tiers.vram < cfg.layers && is_w4a4_preswizzled {
            let max_slot_bytes = (block_bytes + 4096).next_multiple_of(4096);
            log::info(&format!(
                "StreamingEngine: initializing 3-tier streaming ({} VRAM, {} RAM, {} Disk, slot {} MB)",
                tiers.vram, tiers.ram, tiers.disk, max_slot_bytes / 1024 / 1024
            ));
            Some(crate::streaming::StreamingEngine::new(
                &f,
                dev,
                cfg.layers,
                tiers.vram,
                tiers.ram,
                max_slot_bytes,
            )?)
        } else {
            None
        };
        log::info(&format!(
            "DiT 3-Tier Allocation: {} VRAM ({:.2} GB) | {} RAM ({:.2} GB) | {} Disk ({:.2} GB) [Total {} blocks]",
            tiers.vram, (tiers.vram * block_bytes) as f64 / 1e9,
            tiers.ram, (tiers.ram * block_bytes) as f64 / 1e9,
            tiers.disk, (tiers.disk * block_bytes) as f64 / 1e9,
            cfg.layers
        ));
        let mut blocks = vec![];
        for i in 0..cfg.layers {
            if std::env::var("H3_DEBUG").is_ok() {
                let (free, _) = crate::config::vram_info().unwrap_or((0, 0));
                log::info(&format!("loading block {i}, free VRAM {:.2} GB", free as f64 / 1e9));
            }
            let p = format!("blocks.{i}");
            let is_fused = f.has(&format!("{p}.attn.qkv_proj.weight"))
                && f.has(&format!("{p}.attn.out_proj.weight"))
                && f.has(&format!("{p}.mlp.fc1.weight"))
                && f.has(&format!("{p}.mlp.fc2.weight"));
            let place = if i < tiers.vram || !is_fused {
                Placement::Gpu
            } else if i < tiers.vram + tiers.ram {
                Placement::Ram
            } else {
                Placement::Disk
            };
            let ad = if cfg.curve_grid.is_some() { dense_f32(&f, &format!("{p}.adaln_proj.linear"), dev)? } else {
                QLinear::load(&f, &format!("{p}.adaln_proj.linear"), place, dev, dtype)?
            };
            let qkv = if f.has(&format!("{p}.attn.qkv_proj.weight")) {
                QkvProj::Fused(QLinear::load(&f, &format!("{p}.attn.qkv_proj"), place, dev, dtype)?)
            } else {
                let q = LowRankLinear {
                    down: QLinear::load(&f, &format!("{p}.attn.q_proj.down"), place, dev, dtype)?,
                    up: QLinear::load(&f, &format!("{p}.attn.q_proj.up"), place, dev, dtype)?,
                };
                let k = LowRankLinear {
                    down: QLinear::load(&f, &format!("{p}.attn.k_proj.down"), place, dev, dtype)?,
                    up: QLinear::load(&f, &format!("{p}.attn.k_proj.up"), place, dev, dtype)?,
                };
                let v = LowRankLinear {
                    down: QLinear::load(&f, &format!("{p}.attn.v_proj.down"), place, dev, dtype)?,
                    up: QLinear::load(&f, &format!("{p}.attn.v_proj.up"), place, dev, dtype)?,
                };
                QkvProj::Factorized { q, k, v }
            };
            let out = if f.has(&format!("{p}.attn.out_proj.weight")) {
                LinProj::Single(QLinear::load(&f, &format!("{p}.attn.out_proj"), place, dev, dtype)?)
            } else {
                LinProj::Factorized(LowRankLinear {
                    down: QLinear::load(&f, &format!("{p}.attn.out_proj.down"), place, dev, dtype)?,
                    up: QLinear::load(&f, &format!("{p}.attn.out_proj.up"), place, dev, dtype)?,
                })
            };
            let fc1 = if f.has(&format!("{p}.mlp.fc1.weight")) {
                LinProj::Single(QLinear::load(&f, &format!("{p}.mlp.fc1"), place, dev, dtype)?)
            } else {
                LinProj::Factorized(LowRankLinear {
                    down: QLinear::load(&f, &format!("{p}.mlp.fc1.down"), place, dev, dtype)?,
                    up: QLinear::load(&f, &format!("{p}.mlp.fc1.up"), place, dev, dtype)?,
                })
            };
            let fc2 = if f.has(&format!("{p}.mlp.fc2.weight")) {
                LinProj::Single(QLinear::load(&f, &format!("{p}.mlp.fc2"), place, dev, dtype)?)
            } else {
                LinProj::Factorized(LowRankLinear {
                    down: QLinear::load(&f, &format!("{p}.mlp.fc2.down"), place, dev, dtype)?,
                    up: QLinear::load(&f, &format!("{p}.mlp.fc2.up"), place, dev, dtype)?,
                })
            };
            blocks.push(Block {
                norm1: ld(&f, &format!("{p}.norm1.weight"), dev, dtype)?,
                norm2: ld(&f, &format!("{p}.norm2.weight"), dev, dtype)?,
                q_norm: ld(&f, &format!("{p}.attn.q_norm.weight"), dev, dtype)?,
                k_norm: ld(&f, &format!("{p}.attn.k_norm.weight"), dev, dtype)?,
                qkv,
                out,
                fc1,
                fc2,
                adaln: ad,
            });
        }
        let ref_place = if resident == cfg.layers { Placement::Gpu } else { mem.offload };
        let mut refiner = vec![];
        for i in 0..cfg.refiner_layers {
            let p = format!("token_refiner.blocks.{i}");
            refiner.push(RefinerBlock {
                norm1: ld(&f, &format!("{p}.norm1.weight"), dev, dtype)?,
                norm2: ld(&f, &format!("{p}.norm2.weight"), dev, dtype)?,
                q_norm: ld(&f, &format!("{p}.attn.q_norm.weight"), dev, dtype)?,
                k_norm: ld(&f, &format!("{p}.attn.k_norm.weight"), dev, dtype)?,
                qkv: QLinear::load(&f, &format!("{p}.attn.qkv_proj"), ref_place, dev, dtype)?,
                out: QLinear::load(&f, &format!("{p}.attn.out_proj"), ref_place, dev, dtype)?,
                fc1: QLinear::load(&f, &format!("{p}.mlp.fc1"), ref_place, dev, dtype)?,
                fc2: QLinear::load(&f, &format!("{p}.mlp.fc2"), ref_place, dev, dtype)?,
            });
        }
        let (adaln_t_table, time_embedder) = if cfg.curve_grid.is_some() {
            (Some(ld(&f, "adaln_t_table", dev, DType::F32)?), None)
        } else {
            (None, Some((dense_f32(&f, "time_embedder.proj_in", dev)?, dense_f32(&f, "time_embedder.proj_out", dev)?)))
        };
        let mut dit = Self {
            dev: dev.clone(),
            dtype,
            mem: mem.clone(),
            video_patch_proj: dense_f32(&f, "video_patch_proj", dev)?,
            audio_patch_proj: dense_f32(&f, "audio_patch_proj", dev)?,
            condition_proj: QLinear::load(&f, "condition_proj", Placement::Gpu, dev, dtype)?,
            adaln_t_table,
            time_embedder,
            inv_freq: ld(&f, "rope.inv_freq", dev, DType::F32)?,
            refiner,
            refiner_norm: ld(&f, "token_refiner.final_norm.weight", dev, dtype)?,
            blocks,
            resident,
            base_resident: resident,
            tiers,
            block_bytes,
            final_norm: ld(&f, "final_layer.norm.weight", dev, dtype)?,
            final_adaln: dense_f32(&f, "final_layer.adaln_proj.linear", dev)?,
            video_out: dense_f32(&f, "final_layer.video_out", dev)?,
            audio_out: dense_f32(&f, "final_layer.audio_out", dev)?,
            pdd_heads: cfg.pdd_heads,
            control: None,
            capture_blocks: vec![],
            captured: Default::default(),
            #[cfg(feature = "cuda")]
            streaming,
            cfg,
        };
        if let Some((l, strength)) = lora {
            dit.apply_lora(l, strength)?;
        }
        if lora.is_some() && std::env::var("H3_LORA_RUNTIME").is_err() {
            let t = std::time::Instant::now();
            for b in &mut dit.blocks {
                if let QkvProj::Fused(qkv) = &mut b.qkv { qkv.merge_loras(dev)?; }
                if let LinProj::Single(out) = &mut b.out { out.merge_loras(dev)?; }
                if let LinProj::Single(fc1) = &mut b.fc1 { fc1.merge_loras(dev)?; }
                if let LinProj::Single(fc2) = &mut b.fc2 { fc2.merge_loras(dev)?; }
                b.adaln.merge_loras(dev)?;
            }
            for b in &mut dit.refiner {
                for l in [&mut b.qkv, &mut b.out, &mut b.fc1, &mut b.fc2] {
                    l.merge_loras(dev)?;
                }
            }
            log::info(&format!("LoRA merged into the weights in {:.1}s", t.elapsed().as_secs_f64()));
        }
        if crate::weights::int8_enabled() {
            for b in &mut dit.blocks {
                if let LinProj::Single(fc1) = &mut b.fc1 {
                    fc1.interleave_swiglu()?;
                }
            }
        }
        Ok(dit)
    }

    /// 3-Tier distribution across VRAM, RAM, and Disk.
    pub fn tiers(&self) -> crate::config::TierDistribution {
        self.tiers
    }

    /// Blocks resident in VRAM now.
    pub fn resident_blocks(&self) -> usize {
        self.resident
    }

    /// Blocks resident when the DiT was loaded (the most `set_resident` brings back).
    pub fn base_resident_blocks(&self) -> usize {
        self.base_resident
    }

    /// VRAM held by one block's weights.
    pub fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    /// Keep only the first `n` of the loaded-resident blocks in VRAM: the rest stream from pinned RAM every
    /// step, like the blocks that never fit. Used to open up room for a run with more tokens than the load-time
    /// budget assumed (the second, larger pass of an upscaled request), and to take the blocks back afterwards.
    pub fn set_resident(&mut self, n: usize) -> Result<()> {
        let n = n.min(self.base_resident);
        let dev = self.dev.clone();
        for (i, b) in self.blocks.iter_mut().enumerate().take(self.base_resident) {
            let demote = i >= n;
            if let QkvProj::Fused(qkv) = &mut b.qkv {
                if demote { qkv.weight.demote(&dev)?; } else { qkv.weight.promote(&dev)?; }
            }
            if let LinProj::Single(out) = &mut b.out {
                if demote { out.weight.demote(&dev)?; } else { out.weight.promote(&dev)?; }
            }
            if let LinProj::Single(fc1) = &mut b.fc1 {
                if demote { fc1.weight.demote(&dev)?; } else { fc1.weight.promote(&dev)?; }
            }
            if let LinProj::Single(fc2) = &mut b.fc2 {
                if demote { fc2.weight.demote(&dev)?; } else { fc2.weight.promote(&dev)?; }
            }
        }
        self.resident = n;
        Ok(())
    }

    fn apply_lora(&mut self, lora: &Lora, strength: f64) -> Result<()> {
        let dev = self.dev.clone();
        let mut attached = 0usize;
        let mut skipped = vec![];
        let mut attach = |lin: &mut QLinear, name: &str| -> Result<()> {
            if let Some(d) = lora.delta(name, &dev)? {
                if d.a.dim(1)? != lin.in_features || d.b.dim(0)? != lin.out_features || d.reshape.is_some() {
                    skipped.push(name.to_string());
                    return Ok(());
                }
                lin.loras.push(crate::weights::LoraDelta { a: d.a, b: d.b, scale: d.scale * strength });
                attached += 1;
            }
            Ok(())
        };
        for (i, b) in self.blocks.iter_mut().enumerate() {
            if let QkvProj::Fused(qkv) = &mut b.qkv { attach(qkv, &format!("blocks.{i}.attn.qkv_proj"))?; }
            if let LinProj::Single(out) = &mut b.out { attach(out, &format!("blocks.{i}.attn.out_proj"))?; }
            if let LinProj::Single(fc1) = &mut b.fc1 { attach(fc1, &format!("blocks.{i}.mlp.fc1"))?; }
            if let LinProj::Single(fc2) = &mut b.fc2 { attach(fc2, &format!("blocks.{i}.mlp.fc2"))?; }
            attach(&mut b.adaln, &format!("blocks.{i}.adaln_proj.linear"))?;
        }
        for (i, b) in self.refiner.iter_mut().enumerate() {
            attach(&mut b.qkv, &format!("token_refiner.blocks.{i}.attn.qkv_proj"))?;
            attach(&mut b.out, &format!("token_refiner.blocks.{i}.attn.out_proj"))?;
            attach(&mut b.fc1, &format!("token_refiner.blocks.{i}.mlp.fc1"))?;
            attach(&mut b.fc2, &format!("token_refiner.blocks.{i}.mlp.fc2"))?;
        }
        attach(&mut self.final_adaln, "final_layer.adaln_proj.linear")?;
        // output heads: merged (pad-and-add PDD head banks)
        for (lin, name) in [(&mut self.video_out, "final_layer.video_out"), (&mut self.audio_out, "final_layer.audio_out")] {
            let w = lin.weight.get(&dev)?.to_dtype(DType::F32)?;
            let b = lin.bias.clone().context("output head bias")?;
            let (w2, b2) = lora.merge_padded(name, &w, &b, strength, &dev)?;
            if w2.dim(0)? != w.dim(0)? {
                self.pdd_heads = w2.dim(0)? / lin.out_features;
            }
            lin.weight = Held::from_tensor(w2);
            lin.bias = Some(b2);
        }
        if !skipped.is_empty() {
            log::warn(&format!(
                "LoRA: {} targets skipped (shape mismatch with this checkpoint, e.g. {}); {} attached",
                skipped.len(),
                skipped[0],
                attached
            ));
        } else {
            log::info(&format!("LoRA: {attached} layers attached, PDD heads: {}", self.pdd_heads));
        }
        Ok(())
    }

    pub fn audio_scale(&self) -> f64 {
        self.cfg.shift_video / self.cfg.shift_audio
    }

    /// condition_proj + token refiner, once per run.
    pub fn refine_text(&self, text: &Tensor) -> Result<Tensor> {
        Ok(self.refine_text_stages(text)?.pop().unwrap())
    }

    /// Refiner with intermediates: [condition_proj, (attn residual, block out) per block, final].
    pub fn refine_text_stages(&self, text: &Tensor) -> Result<Vec<Tensor>> {
        let mut stages = vec![];
        let x = text.to_dtype(self.dtype)?.to_device(&self.dev)?;
        if x.dim(D::Minus1)? == self.cfg.hidden {
            return Ok(vec![x]);
        }
        let mut x = self.condition_proj.forward(&x)?;
        stages.push(x.clone());
        let (h, d) = (self.cfg.heads, self.cfg.head_dim);
        for b in &self.refiner {
            let n1 = ops::rms_norm(&x, Some(&b.norm1), self.cfg.eps)?;
            let qkv = b.qkv.forward(&n1)?;
            let a = self.attend(&qkv, Some((&b.q_norm, &b.k_norm)), None, 0, h, d)?;
            x = (b.out.forward(&a)? + &x)?;
            stages.push(x.clone());
            let n2 = ops::rms_norm(&x, Some(&b.norm2), self.cfg.eps)?;
            x = (b.fc2.forward(&ops::swiglu(&b.fc1.forward(&n2)?)?)? + &x)?;
            stages.push(x.clone());
        }
        stages.push(ops::rms_norm(&x, Some(&self.refiner_norm), self.cfg.eps)?);
        Ok(stages)
    }

    /// q/k norm (+ rope) out of a fused qkv `[S, 3*H*D]`, attention, `[S, H*D]`.
    fn attend(&self, qkv: &Tensor, norms: Option<(&Tensor, &Tensor)>, rope: Option<&Tensor>, rot: usize, h: usize, d: usize) -> Result<Tensor> {
        let s = qkv.dim(0)?;
        let inner = h * d;
        let (qn, kn) = match norms {
            Some((q, k)) => (Some((Some(q), self.cfg.eps)), Some((Some(k), self.cfg.eps))),
            None => (None, None),
        };
        let q = kernels::qk_norm_rope(qkv, 0, h, d, qn, rope, rot, self.dtype)?;
        let k = kernels::qk_norm_rope(qkv, inner, h, d, kn, rope, rot, self.dtype)?;

        let prefer_flash = std::env::var("H3_ATTN").map(|v| v != "int8" && v != "attn8").unwrap_or(true);
        #[cfg(feature = "flash-attn")]
        {
            if prefer_flash && q.device().is_cuda() && matches!(q.dtype(), DType::F16 | DType::BF16) && d <= 256 {
                let v = qkv.narrow(1, 2 * inner, inner)?.contiguous()?.reshape((s, h, d))?;
                let o = ops::attention(&q.unsqueeze(0)?, &k.unsqueeze(0)?, &v.unsqueeze(0)?, false)?;
                return Ok(o.reshape((s, inner))?);
            }
        }

        if crate::attn8::available(&q) {
            return Ok(crate::attn8::attention_qkv(&q, &k, qkv, 2 * inner)?.reshape((s, inner))?);
        }
        let v = qkv.narrow(1, 2 * inner, inner)?.contiguous()?.reshape((s, h, d))?;
        let o = ops::attention(&q.unsqueeze(0)?, &k.unsqueeze(0)?, &v.unsqueeze(0)?, false)?;
        Ok(o.reshape((s, inner))?)
    }

    /// How many head groups the attention of a block runs in. The fused qkv is 3x a token stream, and with q, k and
    /// the f16 copy of v alive next to it that is most of the memory of a long sequence (an upscaled request's second
    /// pass has ~150k tokens). Head groups project, rope and attend `heads / groups` heads at a time, so the full
    /// qkv never exists. Results are the same; short sequences keep the single fused projection.
    fn attn_groups(&self, tokens: usize, lin: &QLinear) -> usize {
        if let Some(g) = std::env::var("H3_ATTN_GROUPS").ok().and_then(|v| v.parse::<usize>().ok()) {
            return if g >= 1 && self.cfg.heads % g == 0 { g } else { 1 };
        }
        if tokens > 140000 {
            return 28;
        } else if tokens > 85000 {
            return 14;
        } else if tokens > 55000 {
            return 4;
        }
        let qkv_bytes = 3 * tokens * self.cfg.hidden * 2;
        let is_w4a4 = matches!(lin.fmt, crate::weights::QFormat::AmpereW4A4Preswizzled { .. });
        let is_attn8 = crate::attn8::available_arch(&self.dev);
        let ok = (is_w4a4 || is_attn8) && lin.loras.is_empty() && !lin.interleaved && lin.block_scale.is_none() && self.cfg.head_dim == 128;
        let budget: usize = std::env::var("H3_ATTN_BUDGET")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(400_000_000);
        if !ok || qkv_bytes <= budget {
            return 1;
        }
        let want = qkv_bytes.div_ceil(budget);
        let res = (want..=self.cfg.heads).find(|g| self.cfg.heads % g == 0).unwrap_or(self.cfg.heads);
        println!("[attn_groups] is_w4a4={}, is_attn8={}, ok={}, qkv_bytes={}, budget={}, res={}", is_w4a4, is_attn8, ok, qkv_bytes, budget, res);
        res
    }

    /// Attention of one block, `groups` heads groups at a time (see [`attn_groups`](Self::attn_groups)); consumes
    /// `n1`, the modulated input of the qkv projection.
    #[allow(clippy::too_many_arguments)]
    fn attend_factorized(
        &self,
        q: &LowRankLinear,
        k: &LowRankLinear,
        v: &LowRankLinear,
        b: &Block,
        n1: Tensor,
        rope: &Tensor,
        rot: usize,
        groups: usize,
        chunk: usize,
    ) -> Result<AttResult> {
        let (h, d) = (self.cfg.heads, self.cfg.head_dim);
        let (hg, w) = (h / groups, h / groups * d);
        let s = n1.dim(0)?;
        println!("[attend_factorized] start: s={}, groups={}, w={}, chunk={}, free VRAM={:.2} GB", s, groups, w, chunk, crate::config::vram_info().unwrap_or((0,0)).0 as f64 / 1e9);

        let inner = h * d;
        let offload_att = s > 140000;
        let att_bytes = if !offload_att { s * inner * 2 } else { 0 };
        let needed_mid_bytes = 3 * s * 768 * 2;
        let (free_vram_start, _) = crate::config::vram_info().unwrap_or((0, 0));
        let offload_mid = s > 140000 || (s > 30000 && free_vram_start < needed_mid_bytes + att_bytes + 500_000_000);
        let (mid_q, mid_k, mid_v) = if offload_mid {
            println!("[attend_factorized] offloading mid_q/k/v to CPU (s={}, free VRAM={:.2} GB, required={:.2} GB)", s, free_vram_start as f64 / 1e9, (needed_mid_bytes + att_bytes) as f64 / 1e9);
            let mid_q_cpu = prof::time("q down", &self.dev, || q.down.forward_chunked(&n1, chunk))?.to_device(&candle_core::Device::Cpu)?;
            let mid_k_cpu = prof::time("k down", &self.dev, || k.down.forward_chunked(&n1, chunk))?.to_device(&candle_core::Device::Cpu)?;
            let mid_v_cpu = prof::time("v down", &self.dev, || v.down.forward_chunked(&n1, chunk))?.to_device(&candle_core::Device::Cpu)?;
            drop(n1);
            let _ = self.dev.synchronize();
            let (free_after_drop, _) = crate::config::vram_info().unwrap_or((0, 0));
            println!("[attend_factorized] dropped n1, free VRAM restored to {:.2} GB, moving mid_q/k/v back to GPU", free_after_drop as f64 / 1e9);
            let mid_q = mid_q_cpu.to_device(&self.dev)?;
            let mid_k = mid_k_cpu.to_device(&self.dev)?;
            let mid_v = mid_v_cpu.to_device(&self.dev)?;
            (mid_q, mid_k, mid_v)
        } else {
            println!("[attend_factorized] KEEPING mid_q/k/v ON GPU (Zero-Copy Fastpath), free VRAM={:.2} GB", free_vram_start as f64 / 1e9);
            let mid_q = prof::time("q down", &self.dev, || q.down.forward_chunked(&n1, chunk))?;
            let mid_k = prof::time("k down", &self.dev, || k.down.forward_chunked(&n1, chunk))?;
            let mid_v = prof::time("v down", &self.dev, || v.down.forward_chunked(&n1, chunk))?;
            drop(n1);
            let _ = self.dev.synchronize();
            (mid_q, mid_k, mid_v)
        };
        println!("[attend_factorized] offload_att={} (s={})", offload_att, s);

        let pinned_att = if offload_att {
            println!("[attend_factorized] Using Phase 3.3b Pinned Host Attention Buffer (Zero CPU Cat)");
            Some(crate::pinned_att::get_or_create_pinned_att(s, groups, w)?)
        } else {
            None
        };

        let att_full = if !offload_att {
            println!("[attend_factorized] Preallocating att_full on GPU: [{}, {}]", s, inner);
            Some(Tensor::zeros((s, inner), self.dtype, &self.dev)?)
        } else {
            None
        };

        let mut outs = if offload_att && pinned_att.is_none() { Vec::with_capacity(groups) } else { Vec::new() };
        for g in 0..groups {
            let r0 = g * w;
            let q_up = q.up.rows(r0, w, &self.dev)?;
            let qt = prof::time("q up", &self.dev, || q_up.forward_chunked(&mid_q, chunk))?;
            let q_norm = kernels::qk_norm_rope(&qt, 0, hg, d, Some((Some(&b.q_norm), self.cfg.eps)), Some(rope), rot, self.dtype)?;
            drop(qt);

            let k_up = k.up.rows(r0, w, &self.dev)?;
            let kt = prof::time("k up", &self.dev, || k_up.forward_chunked(&mid_k, chunk))?;
            let k_norm = kernels::qk_norm_rope(&kt, 0, hg, d, Some((Some(&b.k_norm), self.cfg.eps)), Some(rope), rot, self.dtype)?;
            drop(kt);

            let v_up = v.up.rows(r0, w, &self.dev)?;
            let vt = prof::time("v up", &self.dev, || v_up.forward_chunked(&mid_v, chunk))?;

            let o = prof::time("attention", &self.dev, || -> Result<Tensor> {
                let prefer_flash = std::env::var("H3_ATTN").map(|v| v != "int8" && v != "attn8").unwrap_or(true);
                #[cfg(feature = "flash-attn")]
                {
                    if prefer_flash && q_norm.device().is_cuda() && matches!(q_norm.dtype(), DType::F16 | DType::BF16) && d <= 256 {
                        let v_res = vt.contiguous()?.reshape((s, hg, d))?;
                        let o = ops::attention(&q_norm.unsqueeze(0)?, &k_norm.unsqueeze(0)?, &v_res.unsqueeze(0)?, false)?;
                        return Ok(o.reshape((s, w))?);
                    }
                }
                if crate::attn8::available(&q_norm) {
                    return Ok(crate::attn8::attention_qkv(&q_norm, &k_norm, &vt, 0)?.reshape((s, w))?);
                }
                let v_res = vt.contiguous()?.reshape((s, hg, d))?;
                let o = ops::attention(&q_norm.unsqueeze(0)?, &k_norm.unsqueeze(0)?, &v_res.unsqueeze(0)?, false)?;
                Ok(o.reshape((s, w))?)
            })?;
            drop((q_norm, k_norm, vt));
            if let Some(ref full) = att_full {
                kernels::copy_slice_col_inplace(full, &o, g * w)?;
                drop(o);
            } else if let Some(ref pbuf) = pinned_att {
                prof::time("d2h pinned o factorized", &self.dev, || pbuf.write_group_async(g, &o))?;
                drop(o);
            } else {
                outs.push(o.to_device(&Device::Cpu)?);
            }
        }
        drop((mid_q, mid_k, mid_v));
        let _ = self.dev.synchronize();
        if let Some(full) = att_full {
            println!("[attend_factorized] Zero-Copy Fastpath via Inplace Copy: returning full GPU tensor!");
            Ok(AttResult::Gpu(full))
        } else if let Some(pbuf) = pinned_att {
            println!("[attend_factorized] Phase 3.3b: returning PinnedHost Attention Buffer!");
            Ok(AttResult::Pinned(pbuf))
        } else if outs.len() == 1 {
            let o = outs.pop().unwrap();
            let o = if o.device().is_cuda() { o } else { o.to_device(&self.dev)? };
            Ok(AttResult::Gpu(o))
        } else {
            let o_cpu = Tensor::cat(&outs, 1)?;
            drop(outs);
            Ok(AttResult::Gpu(o_cpu))
        }
    }

    fn attend_grouped_qkv(&self, qkv: &QLinear, b: &Block, n1: Tensor, r_qkv: &crate::weights::Raw, rope: &Tensor, rot: usize, groups: usize, chunk: usize) -> Result<AttResult> {
        let (h, d) = (self.cfg.heads, self.cfg.head_dim);
        let inner = h * d;
        let (hg, w) = (h / groups, h / groups * d);
        let s = n1.dim(0)?;
        let proj = |base: usize, g: usize| -> Result<Tensor> {
            let r0 = base + g * w;
            qkv.rows(r0, w, &self.dev)?.forward_raw(&n1, &r_qkv.rows(r0, w)?, chunk)
        };
        let offload_att = s > 140000;
        println!("[attend_grouped_qkv] start: s={}, groups={}, w={}, offload_att={}", s, groups, w, offload_att);
        let pinned_att = if offload_att {
            println!("[attend_grouped_qkv] Using Phase 3.2b-lite Pinned Host Attention Buffer (Zero CPU Cat)");
            Some(crate::pinned_att::get_or_create_pinned_att(s, groups, w)?)
        } else {
            None
        };
        let att_full = if !offload_att {
            println!("[attend_grouped_qkv] Preallocating att_full on GPU: [{}, {}]", s, inner);
            Some(Tensor::zeros((s, inner), self.dtype, &self.dev)?)
        } else {
            None
        };
        let mut outs = if offload_att && pinned_att.is_none() { Vec::with_capacity(groups) } else { Vec::new() };
        for g in 0..groups {
            println!("[attend_grouped_qkv] g={}: projecting q, free VRAM={:.2} GB", g, crate::config::vram_info().unwrap_or((0,0)).0 as f64 / 1e9);
            let qt = prof::time("qkv linear", &self.dev, || proj(0, g))?;
            println!("[attend_grouped_qkv] g={}: q projected, free VRAM={:.2} GB", g, crate::config::vram_info().unwrap_or((0,0)).0 as f64 / 1e9);
            let q = kernels::qk_norm_rope(&qt, 0, hg, d, Some((Some(&b.q_norm), self.cfg.eps)), Some(rope), rot, self.dtype)?;
            drop(qt);
            println!("[attend_grouped_qkv] g={}: q_norm done, free VRAM={:.2} GB", g, crate::config::vram_info().unwrap_or((0,0)).0 as f64 / 1e9);
            let kt = prof::time("qkv linear", &self.dev, || proj(inner, g))?;
            println!("[attend_grouped_qkv] g={}: kt projected, free VRAM={:.2} GB", g, crate::config::vram_info().unwrap_or((0,0)).0 as f64 / 1e9);
            let k = kernels::qk_norm_rope(&kt, 0, hg, d, Some((Some(&b.k_norm), self.cfg.eps)), Some(rope), rot, self.dtype)?;
            drop(kt);
            println!("[attend_grouped_qkv] g={}: k_norm done, free VRAM={:.2} GB", g, crate::config::vram_info().unwrap_or((0,0)).0 as f64 / 1e9);
            let vt = prof::time("qkv linear", &self.dev, || proj(2 * inner, g))?;
            println!("[attend_grouped_qkv] g={}: vt projected, free VRAM={:.2} GB", g, crate::config::vram_info().unwrap_or((0,0)).0 as f64 / 1e9);
            let o = prof::time("attention", &self.dev, || -> Result<Tensor> {
                let prefer_flash = std::env::var("H3_ATTN").map(|v| v != "int8" && v != "attn8").unwrap_or(true);
                #[cfg(feature = "flash-attn")]
                {
                    if prefer_flash && q.device().is_cuda() && matches!(q.dtype(), DType::F16 | DType::BF16) && d <= 256 {
                        let v = vt.contiguous()?.reshape((s, hg, d))?;
                        let o = ops::attention(&q.unsqueeze(0)?, &k.unsqueeze(0)?, &v.unsqueeze(0)?, false)?;
                        return Ok(o.reshape((s, w))?);
                    }
                }
                if crate::attn8::available(&q) {
                    return Ok(crate::attn8::attention_qkv(&q, &k, &vt, 0)?.reshape((s, w))?);
                }
                let v = vt.contiguous()?.reshape((s, hg, d))?;
                let o = ops::attention(&q.unsqueeze(0)?, &k.unsqueeze(0)?, &v.unsqueeze(0)?, false)?;
                Ok(o.reshape((s, w))?)
            })?;
            drop((q, k, vt));
            if let Some(ref full) = att_full {
                kernels::copy_slice_col_inplace(full, &o, g * w)?;
                drop(o);
            } else if let Some(ref pbuf) = pinned_att {
                prof::time("d2h pinned o", &self.dev, || pbuf.write_group_async(g, &o))?;
                drop(o);
            } else {
                outs.push(o.to_device(&Device::Cpu)?);
            }
        }
        println!("[attend_grouped_qkv] loop finished, dropping n1...");
        drop(n1);
        if let Some(full) = att_full {
            println!("[attend_grouped_qkv] Zero-Copy Fastpath via Inplace Copy: returning full GPU tensor!");
            Ok(AttResult::Gpu(full))
        } else if let Some(pbuf) = pinned_att {
            println!("[attend_grouped_qkv] Phase 3.2b-lite: returning PinnedHost Attention Buffer!");
            Ok(AttResult::Pinned(pbuf))
        } else if outs.len() == 1 {
            let o = outs.pop().unwrap();
            let o = if o.device().is_cuda() { o } else { o.to_device(&self.dev)? };
            Ok(AttResult::Gpu(o))
        } else {
            let o_cpu = Tensor::cat(&outs, 1)?;
            drop(outs);
            Ok(AttResult::Gpu(o_cpu))
        }
    }

    fn rope_angles(&self, layout: &PackedLayout) -> Result<Tensor> {
        // [S, 3] positions (f32) * inv_freq per axis -> [S, 48] (t | h | w)
        let inv: Vec<f32> = self.inv_freq.to_vec1()?;
        let nf = inv.len();
        let mut a = Vec::with_capacity(layout.seq_len * 3 * nf);
        for p in &layout.positions {
            for ax in p {
                let v = *ax as f32;
                for f in &inv {
                    a.push(v * f);
                }
            }
        }
        let t = Tensor::from_vec(a, (layout.seq_len, 3 * nf), &self.dev)?;
        // the reference stores the rotation table at compute dtype (bf16)
        let tab = kernels::rope_table(&t)?;
        Ok(tab.to_dtype(self.dtype)?.to_dtype(DType::F32)?)
    }

    fn cond_rows(&self, z: &Tensor, video: bool, aug: f64, seed: u64) -> Result<Tensor> {
        let z = z.to_device(&self.dev)?.to_dtype(DType::F32)?;
        let r = if video { patchify_video(&z)? } else { pack_audio(&z)? };
        if aug < 1.0 {
            let noise = crate::rng::randn(seed, r.dims(), &self.dev)?;
            Ok(((r * aug)? + (noise * (1.0 - aug))?)?)
        } else {
            Ok(r)
        }
    }

    /// Build the per-run state for a target latent shape.
    pub fn prepare(&self, cond: &DitCond, video_shape: &[usize], audio_t: usize, sample_sigmas: Vec<f64>) -> Result<DitRun> {
        let text = self.refine_text(&cond.text)?;
        let text_len = text.dim(0)?;
        let (lt, lh, lw) = (video_shape[2], video_shape[3].div_ceil(2) * 2, video_shape[4].div_ceil(2) * 2);
        let mut kf_specs = vec![];
        let mut vrows = vec![];
        let mut arows = vec![];
        for kf in &cond.keyframes {
            kf_specs.push(KeyframeSpec {
                frame_index: kf.frame_index,
                video_t: kf.video.as_ref().map(|v| v.dims()[2]),
                audio_t: kf.audio.as_ref().map(|a| a.dims()[3]),
            });
            if let Some(v) = &kf.video {
                vrows.push(self.cond_rows(v, true, cond.visual_cond_noise_aug, cond.seed)?);
            }
            if let Some(a) = &kf.audio {
                arows.push(self.cond_rows(a, false, cond.audio_cond_noise_aug, cond.seed + 1)?);
            }
        }
        let mut ref_specs = vec![];
        for r in &cond.refs {
            match r {
                RefCond::Image { latent } => {
                    let d = latent.dims();
                    ref_specs.push(RefSpec::Image { lh: d[3], lw: d[4] });
                    vrows.push(self.cond_rows(latent, true, cond.visual_cond_noise_aug, cond.seed)?);
                }
                RefCond::Audio { latent } => {
                    ref_specs.push(RefSpec::Audio { rt: latent.dims()[3] });
                    arows.push(self.cond_rows(latent, false, cond.audio_cond_noise_aug, cond.seed + 1)?);
                }
                RefCond::Video { latent, audio } => {
                    let d = latent.dims();
                    let rt = audio.as_ref().map(|a| a.dims()[3]).unwrap_or(0);
                    ref_specs.push(RefSpec::Video { lt: d[2], lh: d[3], lw: d[4], rt });
                    vrows.push(self.cond_rows(latent, true, cond.visual_cond_noise_aug, cond.seed)?);
                    if let Some(a) = audio {
                        arows.push(self.cond_rows(a, false, cond.audio_cond_noise_aug, cond.seed + 1)?);
                    }
                }
            }
        }
        let layout = PackedLayout::new(text_len, lt, lh, lw, audio_t, &kf_specs, &ref_specs);
        let rope = self.rope_angles(&layout)?;
        let (vmask_rows, vmask_out, amask_rows, amask_out) = self.mask_rows(cond.denoise.as_ref(), lt, lh, lw, audio_t)?;
        let control = match (&cond.control, &self.control) {
            (Some(c), Some(model)) => Some(model.prepare_run(c, &layout)?),
            (Some(_), None) => bail!("a control hint needs a loaded ControlNet (Dit::set_control)"),
            _ => None,
        };
        if let Some(t) = &cond.tags {
            if t.len() != text_len {
                bail!("token tags ({}) do not match text length ({text_len})", t.len());
            }
        }
        Ok(DitRun {
            rope,
            text,
            tags: cond.tags.clone(),
            cond_video_rows: if vrows.is_empty() { None } else { Some(Tensor::cat(&vrows, 0)?) },
            cond_audio_rows: if arows.is_empty() { None } else { Some(Tensor::cat(&arows, 0)?) },
            vis_aug: cond.visual_cond_noise_aug,
            aud_aug: cond.audio_cond_noise_aug,
            latent_t: lt,
            lat_h: lh,
            lat_w: lw,
            audio_t,
            layout,
            vmask_rows,
            vmask_out,
            amask_rows,
            amask_out,
            control,
            sample_sigmas,
        })
    }

    /// Attach (or drop) a Fun ControlNet; requests with a [`ControlCond`] then run it.
    pub fn set_control(&mut self, control: Option<crate::control::Control>) -> Result<()> {
        if let Some(c) = &control {
            c.check_base(self.cfg.time_embed_dim, self.cfg.curve_grid.is_some())?;
        }
        self.control = control;
        Ok(())
    }

    pub fn has_control(&self) -> bool {
        self.control.is_some()
    }

    /// Checkpoint of the attached ControlNet.
    pub fn control_path(&self) -> Option<&str> {
        self.control.as_ref().map(|c| c.path.as_str())
    }

    /// Whether the weights are all resident (the ControlNet then loads resident too).
    pub fn fully_resident(&self) -> bool {
        self.resident == self.cfg.layers
    }

    /// Row-level views of a [`TokenMasks`]: per-row values for the timesteps, and the velocity scale tensors.
    #[allow(clippy::type_complexity)]
    fn mask_rows(&self, m: Option<&TokenMasks>, lt: usize, lh: usize, lw: usize, audio_t: usize) -> Result<(Option<Vec<f32>>, Option<Tensor>, Option<Vec<f32>>, Option<Tensor>)> {
        let Some(m) = m else { return Ok((None, None, None, None)) };
        let (mut vrows, mut vout) = (None, None);
        if let Some(v) = &m.video {
            if v.len() != lt * lh * lw {
                bail!("video mask has {} cells, the latent {}", v.len(), lt * lh * lw);
            }
            let mut rows = Vec::with_capacity(lt * (lh / 2) * (lw / 2));
            for t in 0..lt {
                for y in 0..lh / 2 {
                    for x in 0..lw / 2 {
                        let at = |dy: usize, dx: usize| v[(t * lh + 2 * y + dy) * lw + 2 * x + dx];
                        rows.push(at(0, 0).max(at(0, 1)).max(at(1, 0)).max(at(1, 1)));
                    }
                }
            }
            vrows = Some(rows);
            vout = Some(Tensor::from_vec(v.clone(), (1, 1, lt, lh, lw), &self.dev)?);
        }
        let (mut arows, mut aout) = (None, None);
        if let Some(a) = &m.audio {
            if a.len() != audio_t {
                bail!("audio mask has {} frames, the latent {audio_t}", a.len());
            }
            arows = Some([a.as_slice(), a.as_slice()].concat());
            aout = Some(Tensor::from_vec([a.as_slice(), a.as_slice()].concat(), (1, 1, 2, audio_t), &self.dev)?);
        }
        Ok((vrows, vout, arows, aout))
    }

    /// One transformer block over the packed stream `x` (shared by the base blocks and the ControlNet's), given its
    /// weights already fetched as `[qkv, out, fc1, fc2]`.
    pub(crate) fn run_block(&self, b: &Block, raws: Option<[crate::weights::Raw; 4]>, mut x: Tensor, cx: &StepCtx) -> Result<Tensor> {
        let dev = &self.dev;
        let (h, d, hidden) = (self.cfg.heads, self.cfg.head_dim, self.cfg.hidden);
        let rot = cx.rope.dim(2)? * 2;
        let chunk = self.mem.token_chunk;
        let te = if cx.curve { cx.t_emb.clone() } else { ops::silu(cx.t_emb)?.to_dtype(self.dtype)? };
        let m = b.adaln.forward(&te)?.to_dtype(DType::F32)?; // [U, 6*hidden*3]
        let m = m.reshape((cx.u * 3, 6 * hidden))?;
        let c = |i: usize| m.narrow(1, i * hidden, hidden).and_then(|t| t.contiguous());
        let (shift_msa, scale_msa, gate_msa) = (c(0)?, c(1)?, c(2)?);
        let (shift_mlp, scale_mlp, gate_mlp) = (c(3)?, c(4)?, c(5)?);

        let (free_vram_rb, _) = crate::config::vram_info().unwrap_or((0, 0));
        let groups = match &b.qkv {
            QkvProj::Fused(qkv) => self.attn_groups(x.dim(0)?, qkv),
            QkvProj::Factorized { .. } => {
                if let Some(g) = std::env::var("H3_ATTN_GROUPS").ok().and_then(|v| v.parse::<usize>().ok()) {
                    if g >= 1 && self.cfg.heads % g == 0 {
                        g
                    } else {
                        self.cfg.heads
                    }
                } else {
                    let tokens = x.dim(0)?;
                    if tokens > 140000 {
                        28 // 12s+: 2 heads per group, single QKV = 280MB, 256-byte aligned
                    } else if tokens > 85000 {
                        14 // 8s: 4 heads per group, single QKV = 372MB, 256-byte aligned
                    } else if tokens > 55000 {
                        4  // 5s: 14 heads per group, single QKV = 860MB, validated 306s
                    } else {
                        4  // 3s: 14 heads per group, validated 137s
                    }
                }
            }
        };
        let (n1, x_saved, mut x_holder, dev_ref) = if x.dim(0)? > 30000 {
            println!("[dit] run_block: offloading x to CPU (IN-PLACE NORM1), free VRAM={:.2} GB", crate::config::vram_info().unwrap_or((0,0)).0 as f64 / 1e9);
            let x_cpu = prof::time("offload x", dev, || x.to_device(&Device::Cpu))?;
            prof::time("norm/mod in-place", dev, || kernels::rms_modulate_inplace(&x, &b.norm1, self.cfg.eps, &scale_msa, &shift_msa, cx.idx))?;
            println!("[dit] run_block: x modulated in-place -> n1, free VRAM={:.2} GB", crate::config::vram_info().unwrap_or((0,0)).0 as f64 / 1e9);
            (x, Some(x_cpu), None, dev)
        } else {
            let mut x_h = Some(x);
            let n1 = {
                let x_ref = x_h.as_ref().unwrap();
                prof::time("norm/mod", dev, || -> Result<Tensor> { Ok(kernels::rms_modulate(x_ref, &b.norm1, self.cfg.eps, &scale_msa, &shift_msa, cx.idx)?) })?
            };
            (n1, None, x_h, dev)
        };
        let att_res = match &b.qkv {
            QkvProj::Fused(qkv) if groups > 1 => {
                let r_qkv = match &raws {
                    Some([r, ..]) => r.clone(),
                    None => qkv.fetch(dev)?,
                };
                self.attend_grouped_qkv(qkv, b, n1, &r_qkv, cx.rope, rot, groups, chunk)?
            }
            QkvProj::Fused(qkv) => {
                let qkv_t = prof::time("qkv linear", dev, || match &raws {
                    Some([r_qkv, ..]) => qkv.forward_raw(&n1, r_qkv, chunk),
                    _ => qkv.forward(&n1),
                })?;
                drop(n1);
                let att = prof::time("attention", dev, || self.attend(&qkv_t, Some((&b.q_norm, &b.k_norm)), Some(cx.rope), rot, h, d))?;
                drop(qkv_t);
                AttResult::Gpu(att)
            }
            QkvProj::Factorized { q, k, v } => {
                self.attend_factorized(q, k, v, b, n1, cx.rope, rot, groups, chunk)?
            }
        };

        let s = match &att_res {
            AttResult::Gpu(t) => t.dim(0)?,
            AttResult::Pinned(p) => p.s,
        };
        let mut x = if let Some(x_cpu) = x_saved {
            let (free_x, _) = crate::config::vram_info().unwrap_or((0, 0));
            println!("[dit] run_block: transferring x from CPU back to GPU, free VRAM={:.2} GB, required={:.2} GB", free_x as f64 / 1e9, (s * 5376 * 2) as f64 / 1e9);
            x_cpu.to_device(dev_ref)?
        } else {
            x_holder.unwrap()
        };

        if let AttResult::Pinned(ref pbuf) = att_res {
            let (free_vram_out, _) = crate::config::vram_info().unwrap_or((0, 0));
            println!("[out linear pinned] start Phase 3.2b-lite in-place: s={}, groups={}, w={}, free VRAM={:.2} GB", s, pbuf.groups, pbuf.w, free_vram_out as f64 / 1e9);
            prof::time("out linear in-place pinned", dev, || -> Result<()> {
                let staging = Tensor::zeros((pbuf.groups, chunk, pbuf.w), self.dtype, dev)?;
                let gathered = Tensor::zeros((chunk, pbuf.groups * pbuf.w), self.dtype, dev)?;
                match (&b.out, &raws) {
                    (LinProj::Single(out), Some([_, r_out, ..])) => {
                        let mut st = 0;
                        while st < s {
                            let n = chunk.min(s - st);
                            let chunk_x = x.narrow(0, st, n)?;
                            let chunk_idx = cx.idx.narrow(0, st, n)?;
                            let (stg, gth) = if n == chunk {
                                (staging.clone(), gathered.clone())
                            } else {
                                (Tensor::zeros((pbuf.groups, n, pbuf.w), self.dtype, dev)?, Tensor::zeros((n, pbuf.groups * pbuf.w), self.dtype, dev)?)
                            };
                            pbuf.fetch_chunk_gathered(st, n, &stg, &gth)?;
                            let yc = out.forward_raw(&gth, r_out, chunk)?;
                            kernels::gate_add_inplace(&chunk_x, &yc, &gate_msa, &chunk_idx)?;
                            st += n;
                        }
                    }
                    (LinProj::Single(out), None) => {
                        let r_out = out.fetch(dev)?;
                        let mut st = 0;
                        while st < s {
                            let n = chunk.min(s - st);
                            let chunk_x = x.narrow(0, st, n)?;
                            let chunk_idx = cx.idx.narrow(0, st, n)?;
                            let (stg, gth) = if n == chunk {
                                (staging.clone(), gathered.clone())
                            } else {
                                (Tensor::zeros((pbuf.groups, n, pbuf.w), self.dtype, dev)?, Tensor::zeros((n, pbuf.groups * pbuf.w), self.dtype, dev)?)
                            };
                            pbuf.fetch_chunk_gathered(st, n, &stg, &gth)?;
                            let yc = out.forward_raw(&gth, &r_out, chunk)?;
                            kernels::gate_add_inplace(&chunk_x, &yc, &gate_msa, &chunk_idx)?;
                            st += n;
                        }
                    }
                    (LinProj::Factorized(lr), _) => {
                        let mut st = 0;
                        while st < s {
                            let n = chunk.min(s - st);
                            let chunk_x = x.narrow(0, st, n)?;
                            let chunk_idx = cx.idx.narrow(0, st, n)?;
                            let (stg, gth) = if n == chunk {
                                (staging.clone(), gathered.clone())
                            } else {
                                (Tensor::zeros((pbuf.groups, n, pbuf.w), self.dtype, dev)?, Tensor::zeros((n, pbuf.groups * pbuf.w), self.dtype, dev)?)
                            };
                            pbuf.fetch_chunk_gathered(st, n, &stg, &gth)?;
                            let mid = lr.down.forward(&gth)?;
                            let yc = lr.up.forward(&mid)?;
                            kernels::gate_add_inplace(&chunk_x, &yc, &gate_msa, &chunk_idx)?;
                            st += n;
                        }
                    }
                    _ => anyhow::bail!("unsupported out projection for pinned att"),
                }
                Ok(())
            })?;
        } else if let AttResult::Gpu(att) = att_res {
            if s > 30000 {
                let (free_vram_out, _) = crate::config::vram_info().unwrap_or((0, 0));
                println!("[out linear] start in-place: s={}, att device={:?}, free VRAM={:.2} GB", s, att.device(), free_vram_out as f64 / 1e9);
                prof::time("out linear in-place", dev, || -> Result<()> {
                    match (&b.out, &raws) {
                        (LinProj::Single(out), Some([_, r_out, ..])) => {
                            let mut st = 0;
                            while st < s {
                                let n = chunk.min(s - st);
                                println!("[out linear in-place] st={}, n={}, free VRAM={:.2} GB", st, n, crate::config::vram_info().unwrap_or((0,0)).0 as f64 / 1e9);
                                println!("[out linear in-place] step 1: narrow");
                                let chunk_x = x.narrow(0, st, n)?;
                                let chunk_idx = cx.idx.narrow(0, st, n)?;
                                let chunk_att_raw = att.narrow(0, st, n)?.contiguous()?;
                                let chunk_att = cpu_chunk_to_device(&chunk_att_raw, dev)?;
                                println!("[out linear in-place] step 3: forward_raw");
                                let yc = out.forward_raw(&chunk_att, r_out, chunk)?;
                                drop(chunk_att);
                                println!("[out linear in-place] step 4: gate_add_inplace");
                                kernels::gate_add_inplace(&chunk_x, &yc, &gate_msa, &chunk_idx)?;
                                drop(yc);
                                println!("[out linear in-place] step 5: done chunk");
                                st += n;
                            }
                        }
                        (LinProj::Single(out), None) => {
                            let r_out = out.fetch(dev)?;
                            let mut st = 0;
                            while st < s {
                                let n = chunk.min(s - st);
                                let chunk_x = x.narrow(0, st, n)?;
                                let chunk_idx = cx.idx.narrow(0, st, n)?;
                                let chunk_att_raw = att.narrow(0, st, n)?.contiguous()?;
                                let chunk_att = cpu_chunk_to_device(&chunk_att_raw, dev)?;
                                let yc = out.forward_raw(&chunk_att, &r_out, chunk)?;
                                drop(chunk_att);
                                kernels::gate_add_inplace(&chunk_x, &yc, &gate_msa, &chunk_idx)?;
                                drop(yc);
                                st += n;
                            }
                        }
                        (LinProj::Factorized(lr), _) => {
                            let mut st = 0;
                            while st < s {
                                let n = chunk.min(s - st);
                                let chunk_x = x.narrow(0, st, n)?;
                                let chunk_idx = cx.idx.narrow(0, st, n)?;
                                let chunk_att_raw = att.narrow(0, st, n)?.contiguous()?;
                                let chunk_att = cpu_chunk_to_device(&chunk_att_raw, dev)?;
                                let mid = lr.down.forward(&chunk_att)?;
                                drop(chunk_att);
                                let yc = lr.up.forward(&mid)?;
                                drop(mid);
                                kernels::gate_add_inplace(&chunk_x, &yc, &gate_msa, &chunk_idx)?;
                                drop(yc);
                                st += n;
                            }
                        }
                        _ => {
                            let yc = b.out.forward(&att)?;
                            kernels::gate_add_inplace(&x, &yc, &gate_msa, cx.idx)?;
                        }
                    }
                    Ok(())
                })?;
                drop(att);
            } else {
                let o = prof::time("out linear", dev, || match (&b.out, &raws) {
                    (LinProj::Single(out), Some([_, r_out, ..])) => out.forward_raw(&att, r_out, chunk),
                    (LinProj::Single(out), None) => {
                        let r_out = out.fetch(dev)?;
                        out.forward_raw(&att, &r_out, chunk)
                    }
                    (LinProj::Factorized(lr), _) => {
                        let mid = lr.down.forward(&att)?;
                        lr.up.forward(&mid)
                    }
                    _ => b.out.forward(&att),
                })?;
                drop(att);
                let o = if o.device().is_cuda() { o } else { o.to_device(dev)? };
                x = prof::time("norm/mod", dev, || kernels::gate_add(&x, &o, &gate_msa, cx.idx))?;
                drop(o);
            }
        }

        let s = x.dim(0)?;
        if s > 30000 {
            let (free_vram_mlp, _) = crate::config::vram_info().unwrap_or((0, 0));
            println!("[dit] run_block: mlp start (IN-PLACE GPU RESIDUAL), free VRAM={:.2} GB", free_vram_mlp as f64 / 1e9);
            prof::time("mlp_inplace", dev, || -> Result<()> {
                let mc = self.mem.mlp_chunk.max(1);
                match (&b.fc1, &b.fc2, &raws) {
                    (LinProj::Single(fc1), LinProj::Single(fc2), Some([_, _, r_fc1, r_fc2])) => {
                        let w1 = prof::time("dequant", dev, || fc1.prepare(r_fc1.clone(), dev, self.dtype))?;
                        let w2 = prof::time("dequant", dev, || fc2.prepare(r_fc2.clone(), dev, self.dtype))?;
                        let mut st = 0;
                        while st < s {
                            let n = mc.min(s - st);
                            let chunk_x = x.narrow(0, st, n)?;
                            let chunk_idx = cx.idx.narrow(0, st, n)?;
                            let chunk_n2 = kernels::rms_modulate(&chunk_x, &b.norm2, self.cfg.eps, &scale_mlp, &shift_mlp, &chunk_idx)?;
                            let hc = fc1.run_swiglu(&chunk_n2, &w1, usize::MAX)?;
                            drop(chunk_n2);
                            let out_chunk = fc2.run(&hc, &w2, usize::MAX)?;
                            drop(hc);
                            kernels::gate_add_inplace(&chunk_x, &out_chunk, &gate_mlp, &chunk_idx)?;
                            drop(out_chunk);
                            st += n;
                        }
                    }
                    (LinProj::Single(fc1), LinProj::Single(fc2), None) => {
                        let r_fc1 = fc1.fetch(dev)?;
                        let r_fc2 = fc2.fetch(dev)?;
                        let w1 = prof::time("dequant", dev, || fc1.prepare(r_fc1, dev, self.dtype))?;
                        let w2 = prof::time("dequant", dev, || fc2.prepare(r_fc2, dev, self.dtype))?;
                        let mut st = 0;
                        while st < s {
                            let n = mc.min(s - st);
                            let chunk_x = x.narrow(0, st, n)?;
                            let chunk_idx = cx.idx.narrow(0, st, n)?;
                            let chunk_n2 = kernels::rms_modulate(&chunk_x, &b.norm2, self.cfg.eps, &scale_mlp, &shift_mlp, &chunk_idx)?;
                            let hc = fc1.run_swiglu(&chunk_n2, &w1, usize::MAX)?;
                            drop(chunk_n2);
                            let out_chunk = fc2.run(&hc, &w2, usize::MAX)?;
                            drop(hc);
                            kernels::gate_add_inplace(&chunk_x, &out_chunk, &gate_mlp, &chunk_idx)?;
                            drop(out_chunk);
                            st += n;
                        }
                    }
                    _ => {
                        let mut st = 0;
                        while st < s {
                            let n = mc.min(s - st);
                            let chunk_x = x.narrow(0, st, n)?;
                            let chunk_idx = cx.idx.narrow(0, st, n)?;
                            let chunk_n2 = kernels::rms_modulate(&chunk_x, &b.norm2, self.cfg.eps, &scale_mlp, &shift_mlp, &chunk_idx)?;
                            let h1 = b.fc1.forward(&chunk_n2)?;
                            drop(chunk_n2);
                            let hc = ops::swiglu(&h1)?;
                            drop(h1);
                            let out_chunk = b.fc2.forward(&hc)?;
                            drop(hc);
                            kernels::gate_add_inplace(&chunk_x, &out_chunk, &gate_mlp, &chunk_idx)?;
                            drop(out_chunk);
                            st += n;
                        }
                    }
                }
                Ok(())
            })?;
            return Ok(x);
        }

        let n2 = prof::time("norm/mod", dev, || -> Result<Tensor> { Ok(kernels::rms_modulate(&x, &b.norm2, self.cfg.eps, &scale_mlp, &shift_mlp, cx.idx)?) })?;
        let (x_saved_mlp, mut x_holder_mlp) = (None::<Tensor>, Some(x));

        let (free_vram_mlp, _) = crate::config::vram_info().unwrap_or((0, 0));
        println!("[dit] run_block: mlp start, free VRAM={:.2} GB", free_vram_mlp as f64 / 1e9);
        let mo = prof::time("mlp", dev, || -> Result<Tensor> {
            match (&b.fc1, &b.fc2, &raws) {
                (LinProj::Single(fc1), LinProj::Single(fc2), Some([_, _, r_fc1, r_fc2])) => {
                    let w1 = prof::time("dequant", dev, || fc1.prepare(r_fc1.clone(), dev, self.dtype))?;
                    let w2 = prof::time("dequant", dev, || fc2.prepare(r_fc2.clone(), dev, self.dtype))?;
                    let mut outs = vec![];
                    let mc = self.mem.mlp_chunk.max(1);
                    let mut st = 0;
                    while st < s {
                        let n = mc.min(s - st);
                        let hc = fc1.run_swiglu(&n2.narrow(0, st, n)?, &w1, usize::MAX)?;
                        let out_chunk = fc2.run(&hc, &w2, usize::MAX)?;
                        drop(hc);
                        if s > 30000 {
                            outs.push(out_chunk.to_device(&Device::Cpu)?);
                        } else {
                            outs.push(out_chunk);
                        }
                        st += n;
                    }
                    drop((w1, w2));
                    if outs.len() == 1 {
                        Ok(outs.pop().unwrap())
                    } else {
                        Ok(Tensor::cat(&outs, 0)?)
                    }
                }
                (LinProj::Single(fc1), LinProj::Single(fc2), None) => {
                    let r_fc1 = fc1.fetch(dev)?;
                    let r_fc2 = fc2.fetch(dev)?;
                    let w1 = prof::time("dequant", dev, || fc1.prepare(r_fc1, dev, self.dtype))?;
                    let w2 = prof::time("dequant", dev, || fc2.prepare(r_fc2, dev, self.dtype))?;
                    let mut outs = vec![];
                    let mc = self.mem.mlp_chunk.max(1);
                    let mut st = 0;
                    while st < s {
                        let n = mc.min(s - st);
                        let hc = fc1.run_swiglu(&n2.narrow(0, st, n)?, &w1, usize::MAX)?;
                        let out_chunk = fc2.run(&hc, &w2, usize::MAX)?;
                        drop(hc);
                        if s > 30000 {
                            outs.push(out_chunk.to_device(&Device::Cpu)?);
                        } else {
                            outs.push(out_chunk);
                        }
                        st += n;
                    }
                    drop((w1, w2));
                    if outs.len() == 1 {
                        Ok(outs.pop().unwrap())
                    } else {
                        Ok(Tensor::cat(&outs, 0)?)
                    }
                }
                _ => {
                    let mut outs = vec![];
                    let mc = self.mem.mlp_chunk.max(1);
                    let mut st = 0;
                    while st < s {
                        let n = mc.min(s - st);
                        let n2_chunk = n2.narrow(0, st, n)?;
                        let h1 = b.fc1.forward(&n2_chunk)?;
                        drop(n2_chunk);
                        let hc = ops::swiglu(&h1)?;
                        drop(h1);
                        let out_chunk = b.fc2.forward(&hc)?;
                        drop(hc);
                        if s > 30000 {
                            outs.push(out_chunk.to_device(&Device::Cpu)?);
                        } else {
                            outs.push(out_chunk);
                        }
                        st += n;
                    }
                    if outs.len() == 1 {
                        Ok(outs.pop().unwrap())
                    } else {
                        Ok(Tensor::cat(&outs, 0)?)
                    }
                }
            }
        })?;
        drop(n2);

        let mo = if mo.device().is_cuda() { mo } else { mo.to_device(dev)? };
        let x = if let Some(x_cpu) = x_saved_mlp {
            x_cpu.to_device(dev)?
        } else {
            x_holder_mlp.take().unwrap()
        };
        let x = kernels::gate_add(&x, &mo, &gate_mlp, cx.idx)?;
        drop(mo);

        Ok(x)
    }

    fn time_embed(&self, t_vals: &[f64]) -> Result<Tensor> {
        let t = Tensor::from_vec(t_vals.iter().map(|v| *v as f32).collect::<Vec<_>>(), t_vals.len(), &self.dev)?;
        if let Some(table) = &self.adaln_t_table {
            let g = table.dim(0)?;
            let mut i0 = vec![];
            let mut fr = vec![];
            for v in t_vals {
                let pos = (*v as f32).clamp(0.0, 1.0) * (g - 1) as f32;
                let i = (pos.floor() as usize).min(g - 2);
                i0.push(i as u32);
                fr.push(pos - i as f32);
            }
            let n = t_vals.len();
            let a = table.index_select(&Tensor::from_vec(i0.clone(), n, &self.dev)?, 0)?;
            let b = table.index_select(&Tensor::from_vec(i0.iter().map(|i| i + 1).collect::<Vec<_>>(), n, &self.dev)?, 0)?;
            let w = Tensor::from_vec(fr, (n, 1), &self.dev)?;
            return Ok((&a + (b - &a)?.broadcast_mul(&w)?)?);
        }
        let (pin, pout) = self.time_embedder.as_ref().unwrap();
        let half = self.cfg.timestep_input_dim / 2;
        let freqs: Vec<f32> = (0..half).map(|i| (-(10000f32.ln()) * i as f32 / half as f32).exp()).collect();
        let freqs = Tensor::from_vec(freqs, (1, half), &self.dev)?;
        let args = t.unsqueeze(1)?.broadcast_mul(&freqs)?;
        let emb = Tensor::cat(&[args.cos()?, args.sin()?], 1)?;
        let h = ops::silu(&pin.forward(&emb)?)?;
        Ok(pout.forward(&h)?.to_dtype(self.dtype)?.to_dtype(DType::F32)?)
    }

    /// One model evaluation. `x_audio` is the sampler's carried audio (scaled onto the video
    /// schedule); the returned audio velocity is in the same carried space. `sigma` is the video
    /// sigma. Returns the flow outputs `[video, audio]` (CONST parameterization).
    pub fn forward(&self, run: &DitRun, x_video: &Tensor, x_audio: &Tensor, sigma: f64) -> Result<(Tensor, Tensor)> {
        let scale = self.audio_scale();
        let (sv, sa) = (self.cfg.shift_video, self.cfg.shift_audio);
        let sigma_v = sigma.max(1e-6);
        let sigma_a = time_shift_sigma(sigma_v, sv, sa);
        let carry = sigma_a / sigma_v;
        let audio_own = (x_audio * carry)?;
        let (v, a) = self.forward_inner(run, x_video, &audio_own, sigma_v, sigma_a)?;
        let a = if scale != 1.0 { ((&audio_own * (1.0 - scale))? + (a * (1.0 + (scale - 1.0) * sigma_a))?)? } else { a };
        Ok((v, a))
    }

    fn forward_inner(&self, run: &DitRun, x_video: &Tensor, x_audio: &Tensor, sigma_v: f64, sigma_a: f64) -> Result<(Tensor, Tensor)> {
        let dev = &self.dev;
        let (_, _, ot, oh, ow) = x_video.dims5()?;
        let xv = x_video.to_dtype(DType::F32)?;
        let xv = if oh % 2 != 0 || ow % 2 != 0 { xv.pad_with_same(3, 0, oh % 2)?.pad_with_same(4, 0, ow % 2)? } else { xv };
        let layout = &run.layout;
        let t_v = 1.0 - sigma_v;
        let t_a = 1.0 - sigma_a;
        let seg_t = |k: Seg| -> f64 {
            match k {
                Seg::Text | Seg::Video => t_v,
                Seg::Audio => t_a,
                Seg::Cond | Seg::RefImg => t_v.max(run.vis_aug),
                Seg::CondAudio | Seg::RefAudio => t_a.max(run.aud_aug),
            }
        };
        // masked rows run at their own strength: a row with mask value m sits at sigma = m * sigma_stream, so its
        // label is 1 - m * sigma, held at the cond timestep for rows that are kept
        let (t_pin_v, t_pin_a) = (t_v.max(VISUAL_COND_TIMESTEP), t_a.max(AUDIO_COND_TIMESTEP));
        let v_rows_t: Option<Vec<f64>> = run.vmask_rows.as_ref().map(|m| m.iter().map(|&v| (1.0 - v as f64 * sigma_v).min(t_pin_v)).collect());
        let a_rows_t: Option<Vec<f64>> = run.amask_rows.as_ref().map(|m| m.iter().map(|&v| (1.0 - v as f64 * sigma_a).min(t_pin_a)).collect());
        let mut unique: Vec<f64> = vec![t_v, t_a];
        for s in &layout.segments {
            unique.push(seg_t(s.2));
        }
        unique.extend(v_rows_t.iter().chain(a_rows_t.iter()).flatten());
        unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique.dedup();
        let t_row = |t: f64| unique.binary_search_by(|u| u.partial_cmp(&t).unwrap()).unwrap() as u32;

        // per-token mod-row index into the [3U, hidden] block tables
        let mut idx = vec![0u32; layout.seq_len];
        for &(a, b, k) in &layout.segments {
            let rows_t = match k {
                Seg::Video => v_rows_t.as_deref(),
                Seg::Audio => a_rows_t.as_deref(),
                _ => None,
            };
            if let Some(rt) = rows_t {
                for (slot, &t) in idx[a..b].iter_mut().zip(rt) {
                    *slot = t_row(t) * 3 + k.tag();
                }
                continue;
            }
            let base = t_row(seg_t(k)) * 3;
            for (i, slot) in idx[a..b].iter_mut().enumerate() {
                *slot = if k == Seg::Text { base + run.tags.as_ref().map(|t| t[i]).unwrap_or(1) } else { base + k.tag() };
            }
        }
        let idx = Tensor::from_vec(idx, layout.seq_len, dev)?;

        // embed + pack
        let video_rows = patchify_video(&xv)?;
        let audio_rows = pack_audio(&x_audio.to_dtype(DType::F32)?)?;
        let all_v = match &run.cond_video_rows {
            Some(c) => Tensor::cat(&[c, &video_rows], 0)?,
            None => video_rows,
        };
        let all_a = match &run.cond_audio_rows {
            Some(c) => Tensor::cat(&[c, &audio_rows], 0)?,
            None => audio_rows,
        };
        let (free_vram, _) = crate::config::vram_info().unwrap_or((0, 0));
        println!("[dit] forward_inner: seq_len={}, free VRAM={:.2} GB, running patch proj...", layout.seq_len, free_vram as f64 / 1e9);
        let on_cpu = layout.seq_len > 30000;
        let chunked_proj = |proj: &QLinear, t: &Tensor| -> Result<Tensor> {
            let rows = t.dim(0)?;
            let chunk_size = 4096;
            let mut outs = vec![];
            let mut st = 0;
            while st < rows {
                let n = chunk_size.min(rows - st);
                let chunk = t.narrow(0, st, n)?;
                let p = proj.forward(&chunk)?.to_dtype(self.dtype)?;
                drop(chunk);
                if on_cpu {
                    outs.push(p.to_device(&Device::Cpu)?);
                } else {
                    outs.push(p);
                }
                st += n;
            }
            if outs.len() == 1 {
                let o = outs.pop().unwrap();
                if on_cpu {
                    if o.device().is_cuda() { Ok(o.to_device(&Device::Cpu)?) } else { Ok(o) }
                } else {
                    if o.device().is_cuda() { Ok(o) } else { Ok(o.to_device(dev)?) }
                }
            } else if on_cpu {
                let cat_cpu = Tensor::cat(&outs, 0)?;
                drop(outs);
                Ok(cat_cpu)
            } else {
                Ok(Tensor::cat(&outs, 0)?)
            }
        };

        let ve = chunked_proj(&self.video_patch_proj, &all_v)?;
        let ae = chunked_proj(&self.audio_patch_proj, &all_a)?;
        drop((all_v, all_a));

        let mut parts = vec![];
        let (mut vo, mut ao) = (0, 0);
        for &(a, b, k) in &layout.segments {
            let n = b - a;
            match k {
                Seg::Text => {
                    let txt = if on_cpu { run.text.to_device(&Device::Cpu)? } else { run.text.clone() };
                    parts.push(txt);
                }
                k if k.is_video() => {
                    parts.push(ve.narrow(0, vo, n)?);
                    vo += n;
                }
                _ => {
                    parts.push(ae.narrow(0, ao, n)?);
                    ao += n;
                }
            }
        }
        drop((ve, ae));
        let x_cpu_or_gpu = Tensor::cat(&parts, 0)?;
        drop(parts);
        let mut x = if on_cpu { x_cpu_or_gpu.to_device(dev)? } else { x_cpu_or_gpu };
        let (free_vram2, _) = crate::config::vram_info().unwrap_or((0, 0));
        println!("[dit] forward_inner: x packed shape {:?}, free VRAM={:.2} GB", x.dims(), free_vram2 as f64 / 1e9);
        // the ControlNet branches off the stream as it enters the first block
        let ctrl = match (&self.control, &run.control) {
            (Some(model), Some(rc)) if rc.cond.sigma_end <= sigma_v && sigma_v <= rc.cond.sigma_start => Some((model, rc)),
            _ => None,
        };
        let pristine = ctrl.map(|_| x.clone());
        let mut cstream: Option<Tensor> = None;

        let t_emb = self.time_embed(&unique)?; // [U, t_dim] f32
        let curve = self.adaln_t_table.is_some();
        let hidden = self.cfg.hidden;
        let cx = StepCtx { t_emb: &t_emb, curve, idx: &idx, rope: &run.rope, u: unique.len() };
        #[cfg(feature = "cuda")]
        if let Some(engine) = &self.streaming {
            let step_id = engine.begin_step_and_prime()?;
            let resident = self.resident;
            // 1. Run VRAM resident blocks [0..resident)
            for bi in 0..resident {
                crate::progress::check()?;
                let b = &self.blocks[bi];
                let raws = match (&b.qkv, &b.out, &b.fc1, &b.fc2) {
                    (QkvProj::Fused(qkv), LinProj::Single(out), LinProj::Single(fc1), LinProj::Single(fc2)) => {
                        Some([qkv.fetch(dev)?, out.fetch(dev)?, fc1.fetch(dev)?, fc2.fetch(dev)?])
                    }
                    _ => None,
                };
                x = self.run_block(b, raws, x, &cx)?;
                if let (Some((model, rc)), Some(ci)) = (ctrl, ctrl.and_then(|(m, _)| m.injection.iter().position(|&l| l == bi))) {
                    if ci == 0 {
                        cstream = Some(model.init_stream(pristine.as_ref().unwrap(), &rc.rows, layout)?);
                    }
                    let (c, skip) = model.step(self, ci, cstream.take().unwrap(), &cx)?;
                    cstream = Some(c);
                    x = (x + skip.to_dtype(self.dtype)?.broadcast_mul(&rc.scale.to_dtype(self.dtype)?)?)?;
                }
                if self.capture_blocks.contains(&bi) {
                    self.captured.borrow_mut().push((bi, x.clone()));
                }
            }

            // 2. Run offloaded blocks [resident..47) with Zero-Allocation Double-Buffered StagingRing
            for bi in resident..self.blocks.len() {
                crate::progress::check()?;
                let b = &self.blocks[bi];

                let raws = if let Some(off_idx) = engine.offload_index_of(bi) {
                    let r = engine.acquire_compute_slot(off_idx)?;
                    Some((off_idx, r))
                } else {
                    None
                };

                let raw_weights = raws.as_ref().map(|(_, r)| r.clone());
                x = self.run_block(b, raw_weights, x, &cx)?;

                if let Some((off_idx, _)) = raws {
                    engine.release_compute_slot_and_prefetch(off_idx, step_id)?;
                }

                if let (Some((model, rc)), Some(ci)) = (ctrl, ctrl.and_then(|(m, _)| m.injection.iter().position(|&l| l == bi))) {
                    if ci == 0 {
                        cstream = Some(model.init_stream(pristine.as_ref().unwrap(), &rc.rows, layout)?);
                    }
                    let (c, skip) = model.step(self, ci, cstream.take().unwrap(), &cx)?;
                    cstream = Some(c);
                    x = (x + skip.to_dtype(self.dtype)?.broadcast_mul(&rc.scale.to_dtype(self.dtype)?)?)?;
                }
                if self.capture_blocks.contains(&bi) {
                    self.captured.borrow_mut().push((bi, x.clone()));
                }
            }
        } else {
            let fetch = |b: &Block| -> Result<Option<[crate::weights::Raw; 4]>> {
                match (&b.qkv, &b.out, &b.fc1, &b.fc2) {
                    (QkvProj::Fused(qkv), LinProj::Single(out), LinProj::Single(fc1), LinProj::Single(fc2)) => {
                        Ok(Some([qkv.fetch(dev)?, out.fetch(dev)?, fc1.fetch(dev)?, fc2.fetch(dev)?]))
                    }
                    _ => Ok(None),
                }
            };
            let mut next = fetch(&self.blocks[0])?;
            let mut prev_marker: Option<crate::weights::Marker> = None;
            for (bi, b) in self.blocks.iter().enumerate() {
                crate::progress::check()?;
                let raws = next.take();
                x = self.run_block(b, raws, x, &cx)?;
                if let (Some((model, rc)), Some(ci)) = (ctrl, ctrl.and_then(|(m, _)| m.injection.iter().position(|&l| l == bi))) {
                    if ci == 0 {
                        cstream = Some(model.init_stream(pristine.as_ref().unwrap(), &rc.rows, layout)?);
                    }
                    let (c, skip) = model.step(self, ci, cstream.take().unwrap(), &cx)?;
                    cstream = Some(c);
                    x = (x + skip.to_dtype(self.dtype)?.broadcast_mul(&rc.scale.to_dtype(self.dtype)?)?)?;
                }
                // block bi is enqueued: once the GPU is past block bi-1, start uploading block bi+1
                if let Some(m) = prev_marker.take() {
                    m.wait();
                }
                prev_marker = Some(crate::weights::Marker::record(dev));
                if bi + 1 < self.blocks.len() {
                    next = fetch(&self.blocks[bi + 1])?;
                }
                if self.capture_blocks.contains(&bi) {
                    self.captured.borrow_mut().push((bi, x.clone()));
                }
            }
        }

        // final layer
        let (free_vram_final, _) = crate::config::vram_info().unwrap_or((0, 0));
        println!("[dit] forward_inner: final layer start, free VRAM={:.2} GB", free_vram_final as f64 / 1e9);

        let (va, vb) = layout.find(Seg::Video);
        let (aa, ab) = layout.find(Seg::Audio);
        let xv = x.narrow(0, va, vb - va)?;
        let xa = x.narrow(0, aa, ab - aa)?;
        drop(x);

        let fm = self.final_adaln.forward(&t_emb)?.to_dtype(DType::F32)?; // [U, 2*hidden]
        let shift = fm.narrow(1, 0, hidden)?;
        let fscale = fm.narrow(1, hidden, hidden)?;

        let project_head_chunked = |t_in: &Tensor, rows_t: Option<&[f64]>, default_row: u32, head: &QLinear| -> Result<Tensor> {
            let total = t_in.dim(0)?;
            let chunk_size = 4096;
            let mut outs = vec![];
            let mut st = 0;
            while st < total {
                let n = chunk_size.min(total - st);
                let chunk_x = t_in.narrow(0, st, n)?;
                let n_chunk = ops::rms_norm(&chunk_x, Some(&self.final_norm), self.cfg.eps)?.to_dtype(DType::F32)?;
                drop(chunk_x);
                let h_chunk = match rows_t {
                    Some(rt) => {
                        let rows = Tensor::from_vec(rt[st..st + n].iter().map(|&t| t_row(t)).collect::<Vec<u32>>(), n, dev)?;
                        let sc = (fscale.contiguous()?.index_select(&rows, 0)? + 1.0)?;
                        let sh = shift.contiguous()?.index_select(&rows, 0)?;
                        (n_chunk.broadcast_mul(&sc)? + sh)?
                    }
                    None => {
                        let sc = (fscale.narrow(0, default_row as usize, 1)? + 1.0)?;
                        let sh = shift.narrow(0, default_row as usize, 1)?;
                        n_chunk.broadcast_mul(&sc)?.broadcast_add(&sh)?
                    }
                };
                let out_chunk = head.forward(&h_chunk)?;
                outs.push(out_chunk);
                st += n;
            }
            if outs.len() == 1 { Ok(outs.pop().unwrap()) } else { Ok(Tensor::cat(&outs, 0)?) }
        };

        let v = project_head_chunked(&xv, v_rows_t.as_deref(), t_row(t_v), &self.video_out)?;
        drop(xv);

        let a = project_head_chunked(&xa, a_rows_t.as_deref(), t_row(t_a), &self.audio_out)?;
        drop(xa);
        let video = unpatchify_video(&v, run.latent_t, run.lat_h / 2, run.lat_w / 2, self.cfg.latents_dim)?;
        let video = video.narrow(2, 0, ot)?.narrow(3, 0, oh)?.narrow(4, 0, ow)?;
        let audio = unpack_audio(&a)?;
        // rows that are partly kept predict at mask * sigma: scale their velocity to match
        let video = match &run.vmask_out {
            Some(m) => video.broadcast_mul(m)?,
            None => video,
        };
        let audio = match &run.amask_out {
            Some(m) => audio.broadcast_mul(m)?,
            None => audio,
        };
        let _ = run.audio_t;
        Ok((video.neg()?, audio.neg()?))
    }
}

/// PDD head bank: row block 0 is a full head, later blocks are offsets from it; a step uses the
/// dt-weighted mean of the heads it spans.
fn pdd_head(head: &QLinear, h: &Tensor, n: usize, start: usize, stop: usize, flow_shift: f64) -> Result<Tensor> {
    let w = head.weight.get(h.device())?.to_dtype(DType::F32)?;
    let b = head.bias.as_ref().unwrap().to_dtype(DType::F32)?;
    let k = w.dim(1)?;
    let rows = w.reshape((n, w.dim(0)? / n, k))?;
    let brows = b.reshape((n, b.dim(0)? / n))?;
    let grid: Vec<f64> = (0..=n).map(|i| 1.0 - i as f64 / n as f64).collect();
    let f: Vec<f64> = grid.iter().map(|g| 1.0 - flow_shift * g / (1.0 + (flow_shift - 1.0) * g)).collect();
    let dt: Vec<f64> = (start..stop).map(|i| f[i + 1] - f[i]).collect();
    let tot: f64 = dt.iter().sum();
    let first = start.max(1);
    let mut wm = rows.get(0)?;
    let mut bm = brows.get(0)?;
    for (j, i) in (first..stop).enumerate() {
        let wt = dt[first - start + j] / tot;
        wm = (wm + (rows.get(i)? * wt)?)?;
        bm = (bm + (brows.get(i)? * wt)?)?;
    }
    Ok(h.matmul(&wm.t()?)?.broadcast_add(&bm)?)
}

pub mod prof {
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    thread_local!(static T: RefCell<BTreeMap<&'static str, f64>> = RefCell::new(BTreeMap::new()));
    pub fn on() -> bool {
        std::env::var("H3_PROFILE").is_ok()
    }
    /// Time `f` (device-synchronized) under `name` when H3_PROFILE is set.
    pub fn time<R>(name: &'static str, dev: &candle_core::Device, f: impl FnOnce() -> R) -> R {
        if !on() {
            return f();
        }
        let _ = dev.synchronize();
        let t = std::time::Instant::now();
        let r = f();
        let _ = dev.synchronize();
        T.with(|m| *m.borrow_mut().entry(name).or_default() += t.elapsed().as_secs_f64());
        r
    }
    pub fn dump() {
        T.with(|m| {
            for (k, v) in m.borrow_mut().iter() {
                eprintln!("[h3] profile {k:12} {:8.1} ms", v * 1e3);
            }
            m.borrow_mut().clear();
        });
    }
}

pub mod log {
    pub fn info(s: &str) {
        eprintln!("[h3] {s}");
    }
    pub fn warn(s: &str) {
        eprintln!("[h3] WARNING: {s}");
    }
}
