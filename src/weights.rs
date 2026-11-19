//! ComfyUI checkpoint access: mmapped safetensors, per-tensor placement (GPU / RAM / disk) and the
//! quantized linear layer used by every model (dense, `int8_tensorwise` + ConvRot, `nvfp4` + AWQ).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use memmap2::Mmap;

use crate::kernels;


#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GgmlType {
    Q4_0    = 2,
    Q8_0    = 8,
    Q2K     = 10,
    Q3K     = 11,
    Q4K     = 12,
    Q5K     = 13,
    Q6K     = 14,
    IQ2XXS  = 16,
    IQ2XS   = 17,
    IQ3XXS  = 18,
    IQ4NL   = 20,
    IQ3S    = 21,
    IQ2S    = 22,
    IQ4XS   = 23,
    F64     = 28,
    IQ1_M   = 29,
    TQ1_0   = 34,
    TQ2_0   = 35,
}

impl GgmlType {
    pub fn from_u32(t: u32) -> Option<Self> {
        match t {
            2  => Some(Self::Q4_0),
            8  => Some(Self::Q8_0),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            16 => Some(Self::IQ2XXS),
            17 => Some(Self::IQ2XS),
            18 => Some(Self::IQ3XXS),
            20 => Some(Self::IQ4NL),
            21 => Some(Self::IQ3S),
            22 => Some(Self::IQ2S),
            23 => Some(Self::IQ4XS),
            28 => Some(Self::F64),
            29 => Some(Self::IQ1_M),
            34 => Some(Self::TQ1_0),
            35 => Some(Self::TQ2_0),
            _  => None,
        }
    }

    pub const fn block_elements(self) -> usize {
        match self {
            Self::F64 => 1,
            Self::Q4_0 | Self::Q8_0 | Self::IQ4NL => 32,
            _ => 256,
        }
    }

    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Q4_0   => 18,
            Self::Q8_0   => 34,
            Self::Q2K    => 84,
            Self::Q3K    => 110,
            Self::Q4K    => 144,
            Self::Q5K    => 176,
            Self::Q6K    => 210,
            Self::IQ2XXS => 66,
            Self::IQ2XS  => 74,
            Self::IQ3XXS => 98,
            Self::IQ4NL  => 18,
            Self::IQ3S   => 110,
            Self::IQ2S   => 82,
            Self::IQ4XS  => 136,
            Self::F64    => 8,
            Self::IQ1_M  => 56,
            Self::TQ1_0  => 34,
            Self::TQ2_0  => 66,
        }
    }

    pub const fn tag(self) -> &'static str {
        match self {
            Self::Q4_0   => "GGUF_Q4_0",
            Self::Q8_0   => "GGUF_Q8_0",
            Self::Q2K    => "GGUF_Q2_K",
            Self::Q3K    => "GGUF_Q3_K",
            Self::Q4K    => "GGUF_Q4_K",
            Self::Q5K    => "GGUF_Q5_K",
            Self::Q6K    => "GGUF_Q6_K",
            Self::IQ2XXS => "GGUF_IQ2_XXS",
            Self::IQ2XS  => "GGUF_IQ2_XS",
            Self::IQ3XXS => "GGUF_IQ3_XXS",
            Self::IQ4NL  => "GGUF_IQ4_NL",
            Self::IQ3S   => "GGUF_IQ3_S",
            Self::IQ2S   => "GGUF_IQ2_S",
            Self::IQ4XS  => "GGUF_IQ4_XS",
            Self::F64    => "GGUF_F64",
            Self::IQ1_M  => "GGUF_IQ1_M",
            Self::TQ1_0  => "GGUF_TQ1_0",
            Self::TQ2_0  => "GGUF_TQ2_0",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "GGUF_Q4_0"   => Some(Self::Q4_0),
            "GGUF_Q8_0"   => Some(Self::Q8_0),
            "GGUF_Q2_K"   => Some(Self::Q2K),
            "GGUF_Q3_K"   => Some(Self::Q3K),
            "GGUF_Q4_K"   => Some(Self::Q4K),
            "GGUF_Q5_K"   => Some(Self::Q5K),
            "GGUF_Q6_K"   => Some(Self::Q6K),
            "GGUF_IQ2_XXS" => Some(Self::IQ2XXS),
            "GGUF_IQ2_XS"  => Some(Self::IQ2XS),
            "GGUF_IQ3_XXS" => Some(Self::IQ3XXS),
            "GGUF_IQ4_NL"  => Some(Self::IQ4NL),
            "GGUF_IQ3_S"   => Some(Self::IQ3S),
            "GGUF_IQ2_S"   => Some(Self::IQ2S),
            "GGUF_IQ4_XS"  => Some(Self::IQ4XS),
            "GGUF_F64"     => Some(Self::F64),
            "GGUF_IQ1_M"   => Some(Self::IQ1_M),
            "GGUF_TQ1_0"   => Some(Self::TQ1_0),
            "GGUF_TQ2_0"   => Some(Self::TQ2_0),
            _ => None,
        }
    }
}

/// 雙槽 Ping-Pong 零分配反量化 Arena
struct PingPongArena {
    slots: [Option<Tensor>; 2],
    cursor: usize,
}

impl Default for PingPongArena {
    fn default() -> Self {
        Self { slots: [None, None], cursor: 0 }
    }
}

thread_local! {
    static DEQUANT_ARENA: std::cell::RefCell<HashMap<(candle_core::DeviceLocation, DType), PingPongArena>> = Default::default();
}

/// 從全局 Arena 檢出形狀為 [rows, cols] 的輸出張量視圖 (O(1) 零分配，消滅 1,128 次 alloc_out)
pub fn checkout_dequant_arena(dev: &Device, dtype: DType, rows: usize, cols: usize) -> Result<Tensor> {
    let needed = rows.checked_mul(cols).ok_or_else(|| candle_core::Error::Msg("Dequant dimension overflow".into()))?;
    if needed == 0 {
        bail!("checkout_dequant_arena: zero-sized tensor is not allowed");
    }
    let key = (dev.location(), dtype);

    DEQUANT_ARENA.with(|cell| {
        let mut map = cell.borrow_mut();
        let arena = map.entry(key).or_default();
        let slot_idx = arena.cursor;
        arena.cursor = (arena.cursor + 1) & 1;

        let need_realloc = match &arena.slots[slot_idx] {
            Some(buf) => buf.elem_count() < needed,
            None => true,
        };

        if need_realloc {
            arena.slots[slot_idx] = None;
            let cap = needed.max(5376 * 21504);
            let new_buf = Tensor::zeros((cap,), dtype, dev)?;
            arena.slots[slot_idx] = Some(new_buf);
        }

        let buf = arena.slots[slot_idx].as_ref().unwrap();
        Ok(buf.narrow(0, 0, needed)?.reshape((rows, cols))?)
    })
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub storage_shape: Vec<usize>,
    pub start: usize,
    pub end: usize,
}

/// A memory-mapped `.safetensors` file. Raw bytes are only read when a tensor is materialized.
pub struct SafeFile {
    pub path: String,
    fh: std::fs::File,
    mmap: Mmap,
    data_start: usize,
    pub tensors: HashMap<String, TensorInfo>,
    pub metadata: HashMap<String, String>,
}


fn read_gguf_val(mmap: &[u8], mut pos: usize, val_type: u32) -> Result<(String, usize)> {
    match val_type {
        0 => Ok((mmap[pos].to_string(), pos + 1)),
        1 => Ok(((mmap[pos] as i8).to_string(), pos + 1)),
        2 => {
            let v = u16::from_le_bytes(mmap[pos..pos+2].try_into()?);
            Ok((v.to_string(), pos + 2))
        }
        3 => {
            let v = i16::from_le_bytes(mmap[pos..pos+2].try_into()?);
            Ok((v.to_string(), pos + 2))
        }
        4 => {
            let v = u32::from_le_bytes(mmap[pos..pos+4].try_into()?);
            Ok((v.to_string(), pos + 4))
        }
        5 => {
            let v = i32::from_le_bytes(mmap[pos..pos+4].try_into()?);
            Ok((v.to_string(), pos + 4))
        }
        6 => {
            let v = f32::from_le_bytes(mmap[pos..pos+4].try_into()?);
            Ok((v.to_string(), pos + 4))
        }
        7 => {
            let v = mmap[pos] != 0;
            Ok((v.to_string(), pos + 1))
        }
        8 => {
            let len = u64::from_le_bytes(mmap[pos..pos+8].try_into()?) as usize;
            pos += 8;
            let s = std::str::from_utf8(&mmap[pos..pos+len])?.to_string();
            Ok((s, pos + len))
        }
        9 => {
            let elem_type = u32::from_le_bytes(mmap[pos..pos+4].try_into()?);
            pos += 4;
            let len = u64::from_le_bytes(mmap[pos..pos+8].try_into()?) as usize;
            pos += 8;
            let mut items = Vec::with_capacity(len);
            for _ in 0..len {
                let (item_str, new_p) = read_gguf_val(mmap, pos, elem_type)?;
                items.push(item_str);
                pos = new_p;
            }
            Ok((format!("[{}]", items.join(", ")), pos))
        }
        10 => {
            let v = u64::from_le_bytes(mmap[pos..pos+8].try_into()?);
            Ok((v.to_string(), pos + 8))
        }
        11 => {
            let v = i64::from_le_bytes(mmap[pos..pos+8].try_into()?);
            Ok((v.to_string(), pos + 8))
        }
        12 => {
            let v = f64::from_le_bytes(mmap[pos..pos+8].try_into()?);
            Ok((v.to_string(), pos + 8))
        }
        t => bail!("unsupported gguf val_type {t}"),
    }
}

fn parse_gguf(mmap: &[u8]) -> Result<(usize, HashMap<String, TensorInfo>, HashMap<String, String>)> {
    if mmap.len() < 24 || &mmap[..4] != b"GGUF" {
        bail!("not a valid GGUF file");
    }
    let _version = u32::from_le_bytes(mmap[4..8].try_into()?);
    let tensor_count = u64::from_le_bytes(mmap[8..16].try_into()?) as usize;
    let metadata_kv_count = u64::from_le_bytes(mmap[16..24].try_into()?) as usize;

    let mut pos = 24;
    let mut metadata = HashMap::new();
    let mut alignment = 32usize;

    for _ in 0..metadata_kv_count {
        let key_len = u64::from_le_bytes(mmap[pos..pos+8].try_into()?) as usize;
        pos += 8;
        let key = std::str::from_utf8(&mmap[pos..pos+key_len])?.to_string();
        pos += key_len;
        let val_type = u32::from_le_bytes(mmap[pos..pos+4].try_into()?);
        pos += 4;
        let (val_str, new_pos) = read_gguf_val(mmap, pos, val_type)?;
        pos = new_pos;
        if key == "general.alignment" {
            if let Ok(a) = val_str.parse::<usize>() {
                alignment = a;
            }
        }
        metadata.insert(key, val_str);
    }

    let mut tensors = HashMap::new();
    for _ in 0..tensor_count {
        let name_len = u64::from_le_bytes(mmap[pos..pos+8].try_into()?) as usize;
        pos += 8;
        let name = std::str::from_utf8(&mmap[pos..pos+name_len])?.to_string();
        pos += name_len;
        let n_dims = u32::from_le_bytes(mmap[pos..pos+4].try_into()?);
        pos += 4;
        let mut dims = Vec::with_capacity(n_dims as usize);
        let mut num_elements = 1usize;
        for _ in 0..n_dims {
            let d = u64::from_le_bytes(mmap[pos..pos+8].try_into()?) as usize;
            pos += 8;
            dims.push(d);
            num_elements *= d;
        }
        dims.reverse();

        let type_id = u32::from_le_bytes(mmap[pos..pos+4].try_into()?);
        pos += 4;
        let offset = u64::from_le_bytes(mmap[pos..pos+8].try_into()?) as usize;
        pos += 8;

        let (dtype_str, byte_size, storage_shape) = match type_id {
            0 => ("F32".to_string(), num_elements * 4, dims.clone()),
            1 => ("F16".to_string(), num_elements * 2, dims.clone()),
            24 => ("I8".to_string(),  num_elements * 1, dims.clone()),
            25 => ("I16".to_string(), num_elements * 2, dims.clone()),
            26 => ("I32".to_string(), num_elements * 4, dims.clone()),
            27 => ("I64".to_string(), num_elements * 8, dims.clone()),
            28 => ("F64".to_string(), num_elements * 8, dims.clone()),
            29 | 30 => ("BF16".to_string(), num_elements * 2, dims.clone()),
            t if GgmlType::from_u32(t).is_some() => {
                let gt = GgmlType::from_u32(t).unwrap();
                let blk_elems = gt.block_elements();
                let blk_bytes = gt.block_bytes();
                if num_elements % blk_elems != 0 {
                    bail!("GGUF tensor {name} elements {num_elements} not divisible by {blk_elems}");
                }
                let total_bytes = (num_elements / blk_elems) * blk_bytes;
                let mut s_dims = dims.clone();
                if s_dims.len() >= 2 {
                    let k = s_dims.pop().unwrap();
                    let row_bytes = (k / blk_elems) * blk_bytes;
                    s_dims.push(row_bytes);
                } else {
                    s_dims = vec![total_bytes];
                }
                (gt.tag().to_string(), total_bytes, s_dims)
            }
            t => bail!("unsupported gguf tensor type {t}"),
        };

        let name_stripped = name.strip_prefix("model.diffusion_model.").unwrap_or(&name); let k = name_stripped.strip_prefix("first_stage_model.").map(str::to_string).unwrap_or_else(|| name_stripped.to_string());
        tensors.insert(k, TensorInfo {
            dtype: dtype_str,
            shape: dims,
            storage_shape,
            start: offset,
            end: offset + byte_size,
        });
    }

    let data_start = (pos + alignment - 1) / alignment * alignment;
    Ok((data_start, tensors, metadata))
}
impl SafeFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Arc<Self>> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() >= 4 && &mmap[..4] == b"GGUF" {
            let (data_start, tensors, metadata) = parse_gguf(&mmap)?;
            return Ok(Arc::new(Self { path: path.display().to_string(), fh: file, mmap, data_start, tensors, metadata }));
        }
        let n = u64::from_le_bytes(mmap[..8].try_into()?) as usize;
        let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&mmap[8..8 + n])?;
        let mut tensors = HashMap::new();
        let mut metadata = HashMap::new();
        for (k, v) in header {
            if k == "__metadata__" {
                if let Some(m) = v.as_object() {
                    for (mk, mv) in m {
                        metadata.insert(mk.clone(), mv.as_str().map(str::to_string).unwrap_or_else(|| mv.to_string()));
                    }
                }
                continue;
            }
            let dtype = v["dtype"].as_str().unwrap_or_default().to_string();
            let shape: Vec<usize> = v["shape"].as_array().context("shape")?.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
            let offs = v["data_offsets"].as_array().context("offsets")?;
            let (start, end) = (offs[0].as_u64().unwrap_or(0) as usize, offs[1].as_u64().unwrap_or(0) as usize);
            // some exports keep the ComfyUI checkpoint prefix on the diffusion model's tensors
            let k = k.strip_prefix("model.diffusion_model.").map(str::to_string).unwrap_or(k);
            tensors.insert(k, TensorInfo { dtype, shape: shape.clone(), storage_shape: shape, start, end });
        }
        Ok(Arc::new(Self { path: path.display().to_string(), fh: file, mmap, data_start: 8 + n, tensors, metadata }))
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).with_context(|| format!("tensor {name} not found in {}", self.path))
    }

    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let i = self.info(name)?;
        Ok(&self.mmap[self.data_start + i.start..self.data_start + i.end])
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.info(name)?.shape.clone())
    }

    /// Read a tensor's bytes with `pread` into `buf` (large sequential reads, no page faults).
    pub fn read_into(&self, name: &str, buf: &mut [u8]) -> Result<()> {
        use std::os::unix::fs::FileExt;
        let i = self.info(name)?;
        let len = i.end - i.start;
        self.fh.read_exact_at(&mut buf[..len], (self.data_start + i.start) as u64)?;
        Ok(())
    }

    /// Ask the kernel to read a tensor's pages ahead (sequential, asynchronous readahead), so a later
    /// copy doesn't fault them in one page at a time.
    pub fn prefetch(&self, name: &str) {
        if let Ok(i) = self.info(name) {
            let (off, len) = (self.data_start + i.start, i.end - i.start);
            let page = 4096;
            let start = off / page * page;
            let _ = self.mmap.advise_range(memmap2::Advice::WillNeed, start, len + (off - start));
        }
    }

    /// The storage dtype candle uses for a file dtype (int8 and fp8 are kept as raw bytes).
    pub fn storage_dtype(file_dtype: &str) -> Result<DType> {
        Ok(match file_dtype {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "I8" | "U8" | "F8_E4M3" | "BOOL" => DType::U8,
            dt if dt.starts_with("GGUF_") => DType::U8,
            "I64" => DType::I64,
            "I32" => DType::I32,
            "F64" => DType::F64,
            d => bail!("unsupported safetensors dtype {d}"),
        })
    }

    /// Load a tensor in its storage dtype onto `dev`.
    pub fn load(&self, name: &str, dev: &Device) -> Result<Tensor> {
        let i = self.info(name)?;
        let dt = Self::storage_dtype(&i.dtype)?;
        let shape = if i.storage_shape.is_empty() { vec![1] } else { i.storage_shape.clone() };
        Ok(Tensor::from_raw_buffer(self.bytes(name)?, dt, &shape, dev)?)
    }

    /// Load and convert to `dtype` (float tensors only).
    pub fn load_as(&self, name: &str, dev: &Device, dtype: DType) -> Result<Tensor> {
        let info = self.info(name)?; if let Some(gt) = GgmlType::from_tag(&info.dtype) { let raw = self.load(name, dev)?; let rows = info.shape[0]; let cols = if info.shape.len() > 1 { info.shape[1] } else { 1 }; Ok(crate::kernels::dequant_gguf(&raw, gt, rows, cols, dtype)?) } else { Ok(self.load(name, dev)?.to_dtype(dtype)?) }
    }

    #[inline]
    pub fn data_start(&self) -> usize {
        self.data_start
    }

    #[inline]
    pub fn file_handle(&self) -> &std::fs::File {
        &self.fh
    }

    pub fn names_with_prefix(&self, prefix: &str) -> Vec<String> {
        let mut v: Vec<_> = self.tensors.keys().filter(|k| k.starts_with(prefix)).cloned().collect();
        v.sort();
        v
    }

    /// Parse the `<prefix>.comfy_quant` JSON blob, if any.
    pub fn comfy_quant(&self, prefix: &str) -> Result<Option<serde_json::Value>> {
        let key = format!("{prefix}.comfy_quant");
        if self.has(&key) {
            return Ok(Some(serde_json::from_slice(self.bytes(&key)?)?));
        }
        if let Some(fmt) = self.metadata.get("h3.quant.format") {
            if fmt == "ampere_w4a4_group256_preswizzled" {
                let s_name = format!("{prefix}.weight_scale");
                if self.has(&s_name) {
                    return Ok(Some(serde_json::json!({
                        "format": "ampere_w4a4_group256_preswizzled",
                        "convrot": true,
                        "convrot_groupsize": 256,
                        "group_size": 256
                    })));
                }
            }
        }
        if let Some(meta_str) = self.metadata.get("_quantization_metadata") {
            if let Ok(meta) = serde_json::from_str::<serde_json::Value>(meta_str) {
                if let Some(layer_conf) = meta.get("layers").and_then(|l| l.get(prefix)) {
                    return Ok(Some(layer_conf.clone()));
                }
            }
        }
        Ok(None)
    }
}

/// Where a weight lives between uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Placement {
    /// Resident on the GPU for the lifetime of the model.
    Gpu,
    /// Copied into process RAM at load, uploaded to the GPU for every use.
    Ram,
    /// Left in the mmapped file (OS page cache), uploaded for every use. Lowest RAM footprint.
    Disk,
}

impl std::fmt::Display for Placement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Placement::Gpu => "GPU",
            Placement::Ram => "RAM",
            Placement::Disk => "disk",
        })
    }
}

/// A tensor held according to a [`Placement`]; `get` always returns it on the compute device.
#[derive(Clone)]
pub enum Held {
    Dev(Tensor),
    Host(Tensor),
    /// Raw bytes in page-locked RAM, uploaded on the copy stream (overlaps compute).
    Pinned { buf: Arc<PinnedHostSlice<u8>>, shape: Vec<usize> },
    File { file: Arc<SafeFile>, name: String },
}

// ---------------------------------------------------------------------------------------------
// Asynchronous uploads: a second CUDA stream per device. cudarc's event tracking makes kernels on
// candle's stream wait for these copies, so callers just enqueue the next layer's upload before
// the current layer's kernels have finished.

#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::PinnedHostSlice;
#[cfg(feature = "cuda")]
pub use cuda_upload::upload_u8_async;
#[cfg(feature = "cuda")]
use cuda_upload::{pin_bytes, upload_from_file};
#[cfg(feature = "cuda")]
mod cuda_upload {
use candle_core::cuda_backend::cudarc::driver::CudaStream;
use super::*;
use candle_core::cuda_backend::CudaStorage;

fn copy_stream(dev: &candle_core::CudaDevice) -> Result<Arc<CudaStream>> {
    static STREAMS: std::sync::Mutex<Vec<(candle_core::cuda_backend::DeviceId, Arc<CudaStream>)>> = std::sync::Mutex::new(Vec::new());
    let id = dev.id();
    let mut g = STREAMS.lock().unwrap();
    if let Some((_, s)) = g.iter().find(|(o, _)| *o == id) {
        return Ok(s.clone());
    }
    let s = dev.cuda_stream().context().new_stream().map_err(|e| anyhow::anyhow!("new_stream: {e:?}"))?;
    g.push((id, s.clone()));
    Ok(s)
}

/// Hands a prepared storage to candle as the output of a no-op custom op.
struct Adopt(std::sync::Mutex<Option<CudaStorage>>, candle_core::Shape);
impl candle_core::CustomOp1 for Adopt {
    fn name(&self) -> &'static str {
        "adopt"
    }
    fn cpu_fwd(&self, _: &candle_core::CpuStorage, _: &candle_core::Layout) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("adopt: cuda only")
    }
    fn cuda_fwd(&self, _: &CudaStorage, _: &candle_core::Layout) -> candle_core::Result<(CudaStorage, candle_core::Shape)> {
        Ok((self.0.lock().unwrap().take().expect("adopt used once"), self.1.clone()))
    }
}

/// Upload raw u8 bytes on the copy stream; returns a U8 tensor of `shape` on `dev`.
pub fn upload_u8_async<S: candle_core::cuda_backend::cudarc::driver::HostSlice<u8> + ?Sized>(src: &S, shape: &[usize], dev: &Device) -> Result<Tensor> {
    let cd = match dev {
        Device::Cuda(c) => c,
        _ => bail!("async upload needs a CUDA device"),
    };
    let slice = upload_slice(cd, src)?;
    let storage = CudaStorage::wrap_cuda_slice(slice, cd.clone());
    let seed = Tensor::zeros(1, DType::U8, dev)?;
    Ok(seed.apply_op1_no_bwd(&Adopt(std::sync::Mutex::new(Some(storage)), shape.into()))?)
}

/// Disk -> pinned staging buffer (pread) -> GPU. The staging buffer is reused, so the copy is
/// completed before returning; reads run at disk speed instead of page-fault speed.
pub(super) fn upload_from_file(file: &SafeFile, name: &str, shape: &[usize], dev: &Device) -> Result<Tensor> {
    static STAGING: std::sync::Mutex<Option<PinnedHostSlice<u8>>> = std::sync::Mutex::new(None);
    let Device::Cuda(cd) = dev else { bail!("needs a CUDA device") };
    let len = file.bytes(name)?.len();
    let mut g = STAGING.lock().unwrap();
    if g.as_ref().map(|b| b.len() < len).unwrap_or(true) {
        *g = None;
        *g = Some(unsafe { cd.cuda_stream().context().alloc_pinned::<u8>(len.max(64 << 20)) }.map_err(|e| anyhow::anyhow!("pinned staging: {e:?}"))?);
    }
    let buf = g.as_mut().unwrap();
    let host = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().map_err(|e| anyhow::anyhow!("{e:?}"))?, len) };
    file.read_into(name, host)?;
    let view = unsafe { std::slice::from_raw_parts(host.as_ptr(), len) };
    let slice = upload_slice(cd, view)?;
    copy_stream(cd)?.synchronize().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let storage = CudaStorage::wrap_cuda_slice(slice, cd.clone());
    let seed = Tensor::zeros(1, DType::U8, dev)?;
    Ok(seed.apply_op1_no_bwd(&Adopt(std::sync::Mutex::new(Some(storage)), shape.into()))?)
}

/// Allocate on candle's compute stream (so the buffer is freed in order behind every kernel that
/// reads it, including the raw-pointer custom kernels) and copy on the copy stream: the copy waits
/// for the allocation, readers wait for the copy (cudarc's write event on the slice).
fn upload_slice<S: candle_core::cuda_backend::cudarc::driver::HostSlice<u8> + ?Sized>(
    cd: &candle_core::CudaDevice,
    src: &S,
) -> Result<candle_core::cuda_backend::cudarc::driver::CudaSlice<u8>> {
    let main = cd.cuda_stream();
    let copy = copy_stream(cd)?;
    let mut dst = unsafe { main.alloc::<u8>(src.len()) }.map_err(|e| anyhow::anyhow!("alloc: {e:?}"))?;
    let ready = main.record_event(None).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    copy.wait(&ready).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    copy.memcpy_htod(src, &mut dst).map_err(|e| anyhow::anyhow!("upload: {e:?}"))?;
    Ok(dst)
}

pub(super) fn pin_bytes(bytes: &[u8], dev: &Device) -> Option<Arc<PinnedHostSlice<u8>>> {
    let Device::Cuda(cd) = dev else { return None };
    let mut p = unsafe { cd.cuda_stream().context().alloc_pinned::<u8>(bytes.len()) }.ok()?;
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), p.as_mut_ptr().ok()?, bytes.len()) };
    Some(Arc::new(p))
}

}

#[cfg(not(feature = "cuda"))]
fn upload_from_file(file: &SafeFile, name: &str, _: &[usize], dev: &Device) -> Result<Tensor> { file.load(name, dev) }

#[cfg(not(feature = "cuda"))]
fn pin_bytes(_: &[u8], _: &Device) -> Option<Arc<PinnedHostSlice<u8>>> { None }

#[cfg(not(feature = "cuda"))]
#[derive(Clone)]
pub struct PinnedHostSlice<T>(Vec<T>);
#[cfg(not(feature = "cuda"))]
impl<T> PinnedHostSlice<T> {
    fn len(&self) -> usize { self.0.len() }
    fn as_slice(&self) -> Result<&[T]> { Ok(&self.0) }
}
#[cfg(not(feature = "cuda"))]
fn upload_u8_async(src: &PinnedHostSlice<u8>, shape: &[usize], dev: &Device) -> Result<Tensor> { Ok(Tensor::from_slice(&src.0, shape, dev)?) }

impl Held {
    pub fn new(file: &Arc<SafeFile>, name: &str, place: Placement, dev: &Device) -> Result<Self> {
        Ok(match place {
            Placement::Gpu => Held::Dev(file.load(name, dev)?),
            Placement::Ram => {
                let i = file.info(name)?;
                let dt = SafeFile::storage_dtype(&i.dtype)?;
                let pinned = if dt == DType::U8 || dt == DType::I32 || dt == DType::U32 { pin_bytes(file.bytes(name)?, dev) } else { None };
                match pinned {
                    Some(buf) => Held::Pinned { buf, shape: i.storage_shape.clone() },
                    None => Held::Host(file.load(name, &Device::Cpu)?),
                }
            }
            Placement::Disk => {
                file.info(name)?;
                Held::File { file: file.clone(), name: name.to_string() }
            }
        })
    }

    /// Small tensors (norms, biases, scales) are always kept on the device.
    pub fn new_offloaded_stub(file: &Arc<SafeFile>, name: &str) -> Result<Self> {
        file.info(name)?;
        Ok(Held::File {
            file: file.clone(),
            name: name.to_string(),
        })
    }

    pub fn small(file: &Arc<SafeFile>, name: &str, dev: &Device) -> Result<Self> {
        Ok(Held::Dev(file.load(name, dev)?))
    }

    pub fn from_tensor(t: Tensor) -> Self {
        Held::Dev(t)
    }

    pub fn get(&self, dev: &Device) -> Result<Tensor> {
        Ok(match self {
            Held::Dev(t) => t.clone(),
            Held::Host(t) => t.to_device(dev)?,
            Held::Pinned { buf, shape } => upload_u8_async(&**buf, shape, dev)?,
            Held::File { file, name } => {
                let i = file.info(name)?;
                let dt = SafeFile::storage_dtype(&i.dtype)?;
                if (dt == DType::U8 || dt == DType::I32 || dt == DType::U32) && dev.is_cuda() {
                    upload_from_file(file, name, &i.storage_shape, dev)?
                } else {
                    file.load(name, dev)?
                }
            }
        })
    }

    pub fn shape(&self) -> Result<Vec<usize>> {
        Ok(match self {
            Held::Dev(t) | Held::Host(t) => t.dims().to_vec(),
            Held::Pinned { shape, .. } => shape.clone(),
            Held::File { file, name } => file.shape(name)?,
        })
    }

    pub fn bytes(&self) -> usize {
        match self {
            Held::Dev(t) | Held::Host(t) => t.elem_count() * t.dtype().size_in_bytes(),
            Held::Pinned { buf, .. } => buf.len(),
            Held::File { file, name } => file.bytes(name).map(|b| b.len()).unwrap_or(0),
        }
    }

    /// Start reading a disk-held tensor into the page cache (no-op otherwise).
    pub fn prefetch(&self) {
        if let Held::File { file, name } = self {
            file.prefetch(name);
        }
    }

    pub fn on_gpu(&self) -> bool {
        matches!(self, Held::Dev(_))
    }

    /// Move a device-resident byte tensor (an int8 / fp4 weight) to pinned host memory; its VRAM is
    /// released once the last handle to it drops. Other holders are left alone.
    pub fn demote(&mut self, dev: &Device) -> Result<()> {
        if let Held::Dev(t) = self {
            if t.dtype() == DType::U8 {
                let host = t.to_device(&Device::Cpu)?;
                let shape = host.dims().to_vec();
                *self = match pin_bytes(&host.flatten_all()?.to_vec1::<u8>()?, dev) {
                    Some(buf) => Held::Pinned { buf, shape },
                    None => Held::Host(host),
                };
            }
        }
        Ok(())
    }

    /// Upload a host-held tensor back to the device (the inverse of [`demote`](Self::demote)).
    pub fn promote(&mut self, dev: &Device) -> Result<()> {
        if matches!(self, Held::Pinned { .. } | Held::Host(_)) {
            *self = Held::Dev(self.get(dev)?);
        }
        Ok(())
    }
}

#[derive(Clone)]
pub enum QFormat {
    /// Plain floating point weight.
    Dense,
    /// int8 with f32 per-row (or scalar) scale; ConvRot weights are stored Hadamard-rotated per
    /// `group` input channels, so activations get the same rotation online.
    Int8 { convrot: Option<usize> },
    /// NVFP4 (e2m1 + fp8 e4m3 block-16 scales + f32 global), optional AWQ input smoothing.
    Nvfp4,
    /// Asymmetric W4A8 INT8 with Lloyd-Max codebook, group-16 FP8 relative scale, and Hadamard-256 ConvRot.
    AsymW4A8 { convrot: Option<usize>, group_size: usize, codebook: Held },
    /// Symmetric W4A4 INT4 with Group-256 FP16 scale and Hadamard-256 ConvRot.
    SymW4A4 { convrot: Option<usize>, group_size: usize },
    /// Ampere SM86 Preswizzled W4A4 INT4 Tensor Core format (zero dequant, native mma.m16n8k64).
    AmpereW4A4Preswizzled { convrot: Option<usize>, group_size: usize },
    /// Ampere SM86 Preswizzled W4A8 INT8 Tensor Core format (zero dequant, native mma.m16n8k32).
    AmpereW4A8Preswizzled { convrot: Option<usize>, group_size: usize },
    /// HyperFlow Community W4A16 format (4-bit weights with FP16/BF16 scale).
    W4A16 { convrot: Option<usize>, group_size: usize },
    /// GGUF K-Quant and basic quant formats (Phase 5A universal multi-quant dequantization).
    GgufQuant { gtype: GgmlType },
}

/// A low-rank delta `scale * B @ A` added to a linear layer's output.
#[derive(Clone)]
pub struct LoraDelta {
    pub a: Tensor, // [r, in]
    pub b: Tensor, // [out, r]
    pub scale: f64,
}

/// A layer's stored weight tensors on the device (see [`QLinear::fetch`]).
#[derive(Clone)]
pub struct Raw {
    pub w: Tensor,
    pub bs: Option<Tensor>,
}

impl Raw {
    /// Output rows `[r0, r0 + n)` of an int8 or w4a4 weight (pairs with [`QLinear::rows`]).
    pub fn rows(&self, r0: usize, n: usize) -> Result<Raw> {
        if self.bs.is_some() {
            bail!("row slices of block-scaled weights are not supported");
        }
        if self.w.rank() == 4 {
            if r0 % 8 != 0 || n % 8 != 0 {
                bail!("AmpereW4A4Preswizzled row slice requires r0 and n to be multiples of 8 (got r0={r0}, n={n})");
            }
            Ok(Raw { w: self.w.narrow(0, r0 / 8, n / 8)?.contiguous()?, bs: None })
        } else {
            Ok(Raw { w: self.w.narrow(0, r0, n)?.contiguous()?, bs: None })
        }
    }
}

#[derive(Clone)]
pub struct QLinear {
    pub name: String,
    pub fmt: QFormat,
    pub weight: Held,
    pub scale: Option<Held>,
    pub block_scale: Option<Held>,
    pub pre_quant_scale: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub out_features: usize,
    pub in_features: usize,
    pub loras: Vec<LoraDelta>,
    /// Output rows reordered to (gate, up) pairs: `[g0, u0, g1, u1, ...]` (fused swiglu GEMM).
    pub interleaved: bool,
}

/// Normalized regular Hadamard matrix (ConvRot): Kronecker powers of the 4x4 regular Hadamard.
pub fn hadamard(size: usize, dev: &Device, dtype: DType) -> Result<Tensor> {
    let h4: [f32; 16] = [1., 1., 1., -1., 1., 1., -1., 1., 1., -1., 1., 1., -1., 1., 1., 1.];
    let mut h = h4.to_vec();
    let mut n = 4;
    while n < size {
        let mut out = vec![0f32; n * 4 * n * 4];
        for i in 0..n {
            for j in 0..n {
                for a in 0..4 {
                    for b in 0..4 {
                        out[(i * 4 + a) * (n * 4) + j * 4 + b] = h[i * n + j] * h4[a * 4 + b];
                    }
                }
            }
        }
        h = out;
        n *= 4;
    }
    if n != size {
        bail!("ConvRot group size {size} is not a power of 4");
    }
    let norm = (size as f32).sqrt();
    let h: Vec<f32> = h.into_iter().map(|v| v / norm).collect();
    Ok(Tensor::from_vec(h, (size, size), dev)?.to_dtype(dtype)?)
}

thread_local! {
    static HADAMARD: std::cell::RefCell<HashMap<(usize, String), Tensor>> = Default::default();
}

fn cached_hadamard(size: usize, dev: &Device, dtype: DType) -> Result<Tensor> {
    let key = (size, format!("{:?}{:?}", dev.location(), dtype));
    if let Some(h) = HADAMARD.with(|c| c.borrow().get(&key).cloned()) {
        return Ok(h);
    }
    let h = hadamard(size, dev, dtype)?;
    HADAMARD.with(|c| c.borrow_mut().insert(key, h.clone()));
    Ok(h)
}

/// Rotate the last dim of `x` group-wise by the ConvRot Hadamard.
pub fn rotate_activation(x: &Tensor, group: usize) -> Result<Tensor> {
    if x.dim(D::Minus1)? % group != 0 {
        bail!("ConvRot group {group} does not divide {:?}", x.dims());
    }
    let h = cached_hadamard(group, x.device(), x.dtype())?;
    let n = x.elem_count() / group;
    Ok(x.contiguous()?.reshape((n, group))?.matmul(&h)?.reshape(x.shape())?)
}

impl QLinear {
    /// Load `<prefix>.weight` (+ quantization side tensors, + `<prefix>.bias`) from a ComfyUI file.
    pub fn load(file: &Arc<SafeFile>, prefix: &str, place: Placement, dev: &Device, compute: DType) -> Result<Self> {
        let wname = format!("{prefix}.weight");
        let wshape = file.shape(&wname)?;
        let conf = file.comfy_quant(prefix)?;
        let bias_name = format!("{prefix}.bias");
        let bias = if file.has(&bias_name) { Some(file.load_as(&bias_name, dev, DType::F32)?) } else { None };
        let fmt_name = conf.as_ref().and_then(|c| c["format"].as_str().map(str::to_string));
        let (fmt, scale, block_scale, pqs, in_features) = match fmt_name.as_deref() {
            None => {
                let info = file.info(&wname)?;
                if let Some(gt) = GgmlType::from_tag(&info.dtype) {
                    (QFormat::GgufQuant { gtype: gt }, None, None, None, wshape[1])
                } else {
                    (QFormat::Dense, None, None, None, wshape[1])
                }
            }
            Some("int8_tensorwise") => {
                let c = conf.as_ref().unwrap();
                let convrot = if c["convrot"].as_bool().unwrap_or(false) {
                    Some(c["convrot_groupsize"].as_u64().unwrap_or(256) as usize)
                } else {
                    None
                };
                let s = Held::small(file, &format!("{prefix}.weight_scale"), dev)?;
                (QFormat::Int8 { convrot }, Some(s), None, None, wshape[1])
            }
            Some("nvfp4") => {
                let g = Held::small(file, &format!("{prefix}.weight_scale_2"), dev)?;
                let bs = Held::new(file, &format!("{prefix}.weight_scale"), place, dev)?;
                let pq_name = format!("{prefix}.pre_quant_scale");
                let pq = if file.has(&pq_name) { Some(file.load_as(&pq_name, dev, compute)?) } else { None };
                (QFormat::Nvfp4, Some(g), Some(bs), pq, wshape[1] * 2)
            }
            Some("asym_w4a8_int8") => {
                let c = conf.as_ref().unwrap();
                let convrot = if c["convrot"].as_bool().unwrap_or(false) {
                    Some(c["convrot_groupsize"].as_u64().unwrap_or(256) as usize)
                } else {
                    None
                };
                let group_size = c["group_size"].as_u64().unwrap_or(16) as usize;
                let s_channel = Held::small(file, &format!("{prefix}.weight_s_channel"), dev)?;
                let s_rel = Held::new(file, &format!("{prefix}.weight_s_rel"), place, dev)?;
                let codebook = Held::small(file, &format!("{prefix}.weight_codebook"), dev)?;
                (QFormat::AsymW4A8 { convrot, group_size, codebook }, Some(s_channel), Some(s_rel), None, wshape[1] * 2)
            }
            Some("sym_w4a4_convrot") => {
                let c = conf.as_ref().unwrap();
                let convrot = if c["convrot"].as_bool().unwrap_or(false) {
                    Some(c["convrot_groupsize"].as_u64().unwrap_or(256) as usize)
                } else {
                    None
                };
                let group_size = c["group_size"].as_u64().unwrap_or(256) as usize;
                let s = Held::small(file, &format!("{prefix}.weight_scale"), dev)?;
                (QFormat::SymW4A4 { convrot, group_size }, Some(s), None, None, wshape[1] * 2)
            }
            Some("ampere_w4a4_group256_preswizzled") => {
                let convrot = Some(256);
                let group_size = 256;
                let s = Held::small(file, &format!("{prefix}.weight_scale"), dev)?;
                let k_in = wshape[1] * 64;
                (QFormat::AmpereW4A4Preswizzled { convrot, group_size }, Some(s), None, None, k_in)
            }
            Some("ampere_w4a8_preswizzled") | Some("w4a8_preswizzled") | Some("ampere_w4a8") => {
                let c = conf.as_ref();
                let convrot = if c.and_then(|v| v.get("convrot")).and_then(|v| v.as_bool()).unwrap_or(false) {
                    Some(c.and_then(|v| v.get("convrot_groupsize")).and_then(|v| v.as_u64()).unwrap_or(256) as usize)
                } else {
                    None
                };
                let group_size = c.and_then(|v| v.get("group_size")).and_then(|v| v.as_u64()).unwrap_or(16) as usize;
                let s_name = if file.has(&format!("{prefix}.weight_s_channel")) {
                    format!("{prefix}.weight_s_channel")
                } else {
                    format!("{prefix}.weight_scale")
                };
                let s = Held::small(file, &s_name, dev)?;
                (QFormat::AmpereW4A8Preswizzled { convrot, group_size }, Some(s), None, None, wshape[1])
            }
            Some("w4a16") | Some("sym_w4a16") | Some("hyperflow_w4a16") => {
                let c = conf.as_ref();
                let convrot = if c.and_then(|v| v.get("convrot")).and_then(|v| v.as_bool()).unwrap_or(false) {
                    Some(c.and_then(|v| v.get("convrot_groupsize")).and_then(|v| v.as_u64()).unwrap_or(256) as usize)
                } else {
                    None
                };
                let group_size = c.and_then(|v| v.get("group_size")).and_then(|v| v.as_u64()).unwrap_or(256) as usize;
                let s_name = if file.has(&format!("{prefix}.weight_scale")) {
                    format!("{prefix}.weight_scale")
                } else {
                    format!("{prefix}.weight_scales")
                };
                let s = Held::small(file, &s_name, dev)?;
                (QFormat::W4A16 { convrot, group_size }, Some(s), None, None, wshape[1] * 2)
            }
            Some(f) => bail!("{prefix}: unsupported quantization format {f}"),
        };
        let is_w4a4_preswizzled = matches!(fmt, QFormat::AmpereW4A4Preswizzled { .. });
        let is_preswizzled = matches!(fmt, QFormat::AmpereW4A4Preswizzled { .. } | QFormat::AmpereW4A8Preswizzled { .. });
        let out_features = if is_w4a4_preswizzled { wshape[0] * 8 } else { wshape[0] };
        let weight = if matches!(place, Placement::Ram | Placement::Disk) && is_preswizzled {
            Held::new_offloaded_stub(file, &wname)?
        } else {
            Held::new(file, &wname, place, dev)?
        };
        Ok(Self {
            name: prefix.to_string(),
            fmt,
            weight,
            scale,
            block_scale,
            pre_quant_scale: pqs,
            bias,
            out_features,
            in_features,
            loras: vec![],
            interleaved: false,
        })
    }

    /// This layer restricted to output rows `[r0, r0 + n)`: the same linear for a head group of a fused
    /// projection, to run against [`Raw::rows`]. Needs no LoRA deltas and a plain (int8 / dense) weight.
    pub fn rows(&self, r0: usize, n: usize, dev: &Device) -> Result<QLinear> {
        if !self.loras.is_empty() || self.interleaved || self.block_scale.is_some() {
            bail!("{}: cannot slice rows of this layer", self.name);
        }
        let mut l = self.clone();
        l.out_features = n;
        if matches!(self.fmt, QFormat::Dense | QFormat::GgufQuant { .. }) {
            let w = self.weight.get(dev)?;
            let w_slice = w.narrow(0, r0, n)?.contiguous()?;
            l.weight = Held::from_tensor(w_slice);
        }
        if let Some(sc) = &self.scale {
            let t = sc.get(dev)?;
            if t.elem_count() == self.out_features {
                let rows = t.flatten_all()?.narrow(0, r0, n)?.contiguous()?;
                l.scale = Some(Held::from_tensor(if t.rank() == 2 { rows.reshape((n, 1))? } else { rows }));
            } else if t.dim(0)? == self.out_features {
                let rows = t.narrow(0, r0, n)?.contiguous()?;
                l.scale = Some(Held::from_tensor(rows));
            }
        }
        l.bias = self.bias.as_ref().map(|b| b.narrow(0, r0, n).and_then(|t| t.contiguous())).transpose()?;
        Ok(l)
    }

    pub fn weight_bytes(&self) -> usize {
        self.weight.bytes() + self.block_scale.as_ref().map(|b| b.bytes()).unwrap_or(0)
    }

    /// Readahead for disk-held weights (see [`SafeFile::prefetch`]).
    pub fn prefetch(&self) {
        self.weight.prefetch();
        if let Some(b) = &self.block_scale {
            b.prefetch();
        }
    }

    /// Start uploading the stored weight (async for streamed int8 / nvfp4 weights).
    pub fn fetch(&self, dev: &Device) -> Result<Raw> {
        Ok(Raw { w: self.weight.get(dev)?, bs: self.block_scale.as_ref().map(|b| b.get(dev)).transpose()? })
    }

    /// Dequantized `[out, in]` weight in `dtype` on `dev`. For ConvRot weights this is still in the
    /// rotated basis (the caller rotates its activations).
    pub fn dequant(&self, dev: &Device, dtype: DType) -> Result<Tensor> {
        self.dequant_raw(&self.fetch(dev)?, dev, dtype)
    }

    pub fn dequant_raw(&self, raw: &Raw, dev: &Device, dtype: DType) -> Result<Tensor> {
        let w = &raw.w;
        Ok(match &self.fmt {
            QFormat::Dense => w.to_dtype(dtype)?,
            QFormat::GgufQuant { gtype } => {
                kernels::dequant_gguf(w, *gtype, self.out_features, self.in_features, dtype)?
            }
            QFormat::Int8 { .. } => kernels::dequant_i8(w, &self.scale.as_ref().unwrap().get(dev)?, dtype)?,
            QFormat::Nvfp4 => kernels::dequant_nvfp4(w, raw.bs.as_ref().unwrap(), &self.scale.as_ref().unwrap().get(dev)?, dtype)?,
            QFormat::AsymW4A8 { codebook, .. } => {
                let cb = codebook.get(dev)?;
                let s_rel = raw.bs.as_ref().context("AsymW4A8 requires s_rel")?;
                let s_channel = self.scale.as_ref().unwrap().get(dev)?;
                kernels::dequant_w4a8(w, s_rel, &s_channel, &cb, self.in_features, dtype)?
            }
            QFormat::SymW4A4 { .. } => {
                let scale = self.scale.as_ref().unwrap().get(dev)?;
                kernels::dequant_sym_w4a4(w, &scale, dtype)?
            }
            QFormat::AmpereW4A4Preswizzled { .. } => {
                bail!("AmpereW4A4Preswizzled runs natively on INT4 Tensor Cores without dequantization");
            }
            QFormat::AmpereW4A8Preswizzled { .. } => {
                kernels::dequant_i8(w, &self.scale.as_ref().unwrap().get(dev)?, dtype)?
            }
            QFormat::W4A16 { .. } => {
                let scale = self.scale.as_ref().unwrap().get(dev)?;
                kernels::dequant_sym_w4a4(w, &scale, dtype)?
            }
        })
    }

    /// Dequantized weight in the plain (unrotated) basis.
    pub fn dequant_plain(&self, dev: &Device, dtype: DType) -> Result<Tensor> {
        let w = self.dequant(dev, DType::F32)?;
        let w = match &self.fmt {
            QFormat::Int8 { convrot: Some(g) }
            | QFormat::AsymW4A8 { convrot: Some(g), .. }
            | QFormat::SymW4A4 { convrot: Some(g), .. }
            | QFormat::AmpereW4A4Preswizzled { convrot: Some(g), .. }
            | QFormat::AmpereW4A8Preswizzled { convrot: Some(g), .. }
            | QFormat::W4A16 { convrot: Some(g), .. } => rotate_activation(&w, *g)?,
            _ => w,
        };
        Ok(w.to_dtype(dtype)?)
    }

    /// `y = x @ W^T + b` (+ LoRA), processing at most `chunk` rows of `x` per GEMM.
    pub fn forward_chunked(&self, x: &Tensor, chunk: usize) -> Result<Tensor> {
        #[cfg(feature = "mlx")]
        if x.device().is_metal() && matches!(self.fmt, QFormat::Int8 { .. }) && self.in_features % 64 == 0 {
            return self.forward_raw(x, &self.fetch(x.device())?, chunk);
        }
        if x.device().is_cuda() && matches!(self.fmt, QFormat::AmpereW4A4Preswizzled { .. } | QFormat::AmpereW4A8Preswizzled { .. }) {
            return self.forward_raw(x, &self.fetch(x.device())?, chunk);
        }
        let w = self.dequant(x.device(), x.dtype())?;
        self.apply(x, &w, chunk)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_chunked(x, usize::MAX)
    }

    /// Apply with an already dequantized weight (see [`QLinear::dequant`]).
    pub fn forward_chunked_to_device(&self, x: &Tensor, chunk: usize, dst_dev: &Device) -> Result<Tensor> {
        let w = self.dequant(x.device(), x.dtype())?;
        self.apply_to_device(x, &w, chunk, dst_dev)
    }

    pub fn apply(&self, x: &Tensor, w: &Tensor, chunk: usize) -> Result<Tensor> {
        self.apply_to_device(x, w, chunk, x.device())
    }

    pub fn apply_to_device(&self, x: &Tensor, w: &Tensor, chunk: usize, dst_dev: &Device) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let k = *dims.last().unwrap();
        let rows = x.elem_count() / k;
        let x2 = x.reshape((rows, k))?;
        let x2 = match &self.pre_quant_scale {
            Some(s) => x2.broadcast_mul(&s.to_dtype(x.dtype())?)?,
            None => x2,
        };
        let wt = w.t()?;
        let mut outs = vec![];
        let chunk = chunk.max(1);
        let mut start = 0;
        while start < rows {
            let n = chunk.min(rows - start);
            let xc = x2.narrow(0, start, n)?;
            let xr = match self.fmt {
                QFormat::Int8 { convrot: Some(g) }
                | QFormat::AsymW4A8 { convrot: Some(g), .. }
                | QFormat::SymW4A4 { convrot: Some(g), .. }
                | QFormat::AmpereW4A8Preswizzled { convrot: Some(g), .. }
                | QFormat::W4A16 { convrot: Some(g), .. } => rotate_activation(&xc, g)?,
                _ => xc.clone(),
            };
            let mut y = xr.matmul(&wt)?;
            if let Some(b) = &self.bias {
                y = y.broadcast_add(&b.to_dtype(y.dtype())?)?;
            }
            for l in &self.loras {
                let a = l.a.to_dtype(x.dtype())?;
                let b = l.b.to_dtype(x.dtype())?;
                let d = xc.matmul(&a.t()?)?.matmul(&b.t()?)?;
                y = (y + (d * l.scale)?)?;
            }
            let y_dst = if y.device().same_device(dst_dev) { y } else { y.to_device(dst_dev)? };
            outs.push(y_dst);
            start += n;
        }
        let y = if outs.len() == 1 { outs.pop().unwrap() } else { Tensor::cat(&outs, 0)? };
        let mut od = dims;
        *od.last_mut().unwrap() = self.out_features;
        Ok(y.reshape(od)?)
    }
}

/// Load a small dense tensor as `dtype` on `dev`.
pub fn small(file: &SafeFile, name: &str, dev: &Device, dtype: DType) -> Result<Tensor> {
    file.load_as(name, dev, dtype)
}

/// A point on the compute stream the host can wait for (bounds how far uploads run ahead).
#[cfg(feature = "cuda")]
pub struct Marker(Option<candle_core::cuda_backend::cudarc::driver::CudaEvent>);
#[cfg(not(feature = "cuda"))]
pub struct Marker;

#[cfg(feature = "cuda")]
impl Marker {
    pub fn record(dev: &Device) -> Self {
        match dev {
            Device::Cuda(c) => Marker(c.cuda_stream().record_event(None).ok()),
            _ => Marker(None),
        }
    }
    pub fn wait(&self) {
        if let Some(e) = &self.0 {
            let _ = e.synchronize();
        }
    }
}

/// Interleave the two halves of a row dimension: `[a0..aF, b0..bF]` -> `[a0, b0, a1, b1, ...]`.
fn interleave_rows(t: &Tensor) -> Result<Tensor> {
    let n = t.dim(0)?;
    let rest: Vec<usize> = t.dims()[1..].to_vec();
    let mut shape = vec![2, n / 2];
    shape.extend(&rest);
    let mut perm: Vec<usize> = (0..shape.len()).collect();
    perm.swap(0, 1);
    Ok(t.reshape(shape)?.permute(perm)?.contiguous()?.reshape(t.shape())?)
}

impl QLinear {
    /// Fold attached LoRA deltas into the weight (like ComfyUI's LoRA patching of quantized
    /// weights: dequantize, add, requantize), so they cost nothing per step.
    pub fn merge_loras(&mut self, dev: &Device) -> Result<()> {
        if self.loras.is_empty() || self.interleaved {
            return Ok(());
        }
        let delta = |r0: usize, n: usize| -> Result<Tensor> {
            let mut acc: Option<Tensor> = None;
            for l in &self.loras {
                let b = l.b.to_device(dev)?.narrow(0, r0, n)?.to_dtype(DType::F32)?;
                let d = (b.matmul(&l.a.to_device(dev)?.to_dtype(DType::F32)?)? * l.scale)?;
                acc = Some(match acc {
                    None => d,
                    Some(a) => (a + d)?,
                });
            }
            Ok(acc.unwrap())
        };
        match self.fmt {
            QFormat::Nvfp4
            | QFormat::AsymW4A8 { .. }
            | QFormat::SymW4A4 { .. }
            | QFormat::AmpereW4A4Preswizzled { .. }
            | QFormat::AmpereW4A8Preswizzled { .. }
            | QFormat::W4A16 { .. }
            | QFormat::GgufQuant { .. } => return Ok(()),
            QFormat::Dense => {
                let w = self.weight.get(dev)?;
                let dt = w.dtype();
                let merged = (w.to_dtype(DType::F32)? + delta(0, self.out_features)?)?.to_dtype(dt)?;
                self.weight = Held::Dev(merged);
            }
            QFormat::Int8 { convrot } => {
                // row chunks keep the f32 temporaries small (the GPU may be shared)
                let raw = self.weight.get(dev)?;
                let sc = self.scale.as_ref().unwrap().get(dev)?.to_dtype(DType::F32)?;
                let per_row = sc.elem_count() == self.out_features;
                let (mut qs, mut scales) = (vec![], vec![]);
                let step = (1usize << 26) / self.in_features.max(1);
                let mut r0 = 0;
                while r0 < self.out_features {
                    let n = step.min(self.out_features - r0);
                    let s_rows = if per_row { sc.reshape(self.out_features)?.narrow(0, r0, n)?.contiguous()? } else { sc.clone() };
                    let w = kernels::dequant_i8(&raw.narrow(0, r0, n)?.contiguous()?, &s_rows, DType::F32)?;
                    let w = match convrot {
                        Some(g) => rotate_activation(&w, g)?, // back to the plain basis
                        None => w,
                    };
                    let w = (w + delta(r0, n)?)?;
                    let w = match convrot {
                        Some(g) => rotate_activation(&w, g)?,
                        None => w,
                    };
                    let scale = (w.abs()?.max_keepdim(1)? / 127.0)?.clamp(1e-30, f64::MAX)?;
                    let q = w.broadcast_div(&scale)?.round()?.clamp(-128.0, 127.0)?;
                    // int8 two's complement stored as bytes
                    let q = q.ge(0.0)?.where_cond(&q, &(&q + 256.0)?)?.to_dtype(DType::U8)?;
                    qs.push(q.to_device(&Device::Cpu)?);
                    scales.push(scale);
                    r0 += n;
                }
                let host = Tensor::cat(&qs, 0)?;
                self.scale = Some(Held::Dev(Tensor::cat(&scales, 0)?));
                self.weight = match &self.weight {
                    Held::Dev(_) => Held::Dev(host.to_device(dev)?),
                    _ => {
                        let bytes = host.flatten_all()?.to_vec1::<u8>()?;
                        match pin_bytes(&bytes, dev) {
                            Some(buf) => Held::Pinned { buf, shape: host.dims().to_vec() },
                            None => Held::Host(host),
                        }
                    }
                };
            }
        }
        self.loras.clear();
        Ok(())
    }

    /// Reorder a gate/up projection's output rows into pairs so `swiglu` fuses into the int8 GEMM.
    /// Weights streamed from disk are left as they are.
    pub fn interleave_swiglu(&mut self) -> Result<()> {
        if self.interleaved || matches!(self.weight, Held::File { .. }) || !matches!(self.fmt, QFormat::Int8 { .. }) {
            return Ok(());
        }
        let w = match &self.weight {
            Held::Dev(t) => t.to_device(&Device::Cpu)?,
            Held::Host(t) => t.clone(),
            Held::Pinned { buf, shape } => Tensor::from_vec(buf.as_slice().map_err(|e| anyhow::anyhow!("{e:?}"))?.to_vec(), shape.as_slice(), &Device::Cpu)?,
            Held::File { .. } => unreachable!(),
        };
        let wi = interleave_rows(&w)?;
        self.weight = match &self.weight {
            Held::Dev(t) => Held::Dev(wi.to_device(t.device())?),
            Held::Host(_) => Held::Host(wi),
            Held::Pinned { shape, .. } => {
                let bytes = wi.flatten_all()?.to_vec1::<u8>()?;
                let dev = self.scale.as_ref().and_then(|s| if let Held::Dev(t) = s { Some(t.device().clone()) } else { None }).unwrap_or(Device::Cpu);
                match pin_bytes(&bytes, &dev) {
                    Some(buf) => Held::Pinned { buf, shape: shape.clone() },
                    None => Held::Host(wi),
                }
            }
            Held::File { .. } => unreachable!(),
        };
        if let Some(Held::Dev(sc)) = &self.scale {
            if sc.elem_count() == self.out_features {
                self.scale = Some(Held::Dev(interleave_rows(sc)?));
            }
        }
        if let Some(b) = &self.bias {
            self.bias = Some(interleave_rows(b)?);
        }
        for l in &mut self.loras {
            l.b = interleave_rows(&l.b)?;
        }
        self.interleaved = true;
        Ok(())
    }
}

/// `silu(gate) * up` for a `[.., 2F]` tensor in interleaved (`g0, u0, g1, u1, ...`) layout.
pub fn swiglu_interleaved(x: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let f = dims[dims.len() - 1] / 2;
    let rows = x.elem_count() / (2 * f);
    let x = x.reshape((rows, f, 2))?;
    let g = x.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?;
    let u = x.narrow(2, 1, 1)?.squeeze(2)?;
    let mut od = dims;
    *od.last_mut().unwrap() = f;
    Ok((candle_nn::ops::silu(&g)? * u)?.reshape(od)?)
}

/// int8 weights run on int8 tensor cores (cuBLASLt) unless `H3_INT8=0`.
pub fn int8_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("H3_INT8").map(|v| v != "0").unwrap_or(true))
}

pub fn mmq_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("H3_MMQ").map(|v| v != "0").unwrap_or(false))
}

/// A weight ready for GEMMs: raw int8 (tensor-core path) or dequantized.
pub enum Prepared {
    W4A4(Raw),
    Int8(Raw),
    Mmq(Raw),
    #[cfg(feature = "mlx")]
    Mlx(crate::apple::Int8Weight),
    Deq(Tensor),
}

impl QLinear {
    pub fn prepare(&self, raw: Raw, dev: &Device, dtype: DType) -> Result<Prepared> {
        #[cfg(feature = "mlx")]
        if matches!(self.fmt, QFormat::Int8 { .. }) && dev.is_metal() && self.in_features % 64 == 0 {
            return Ok(Prepared::Mlx(crate::apple::Int8Weight::new(&raw.w, &self.scale.as_ref().unwrap().get(dev)?)?));
        }
        Ok(match &self.fmt {
            QFormat::AmpereW4A4Preswizzled { .. } if dev.is_cuda() => Prepared::W4A4(raw),
            QFormat::AmpereW4A8Preswizzled { .. } if dev.is_cuda() => Prepared::Int8(raw),
            QFormat::Int8 { .. } if int8_enabled() && dev.is_cuda() => Prepared::Int8(raw),
            QFormat::GgufQuant { gtype: GgmlType::Q2K | GgmlType::Q3K | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q8_0 | GgmlType::Q4_0 | GgmlType::IQ4XS } if mmq_enabled() && dev.is_cuda() && dtype == DType::BF16 => {
                Prepared::Mmq(raw)
            }
            QFormat::AsymW4A8 { codebook, .. } if int8_enabled() && dev.is_cuda() => {
                let cb = codebook.get(dev)?;
                let s_rel = raw.bs.as_ref().context("AsymW4A8 requires s_rel")?;
                let w_i8 = kernels::dequant_w4a8_to_i8(&raw.w, s_rel, &cb, self.in_features)?;
                Prepared::Int8(Raw { w: w_i8, bs: None })
            }
            _ => Prepared::Deq(self.dequant_raw(&raw, dev, dtype)?),
        })
    }

    /// `forward_chunked` with an already fetched weight.
    pub fn forward_raw(&self, x: &Tensor, raw: &Raw, chunk: usize) -> Result<Tensor> {
        let p = self.prepare(raw.clone(), x.device(), x.dtype())?;
        self.run(x, &p, chunk)
    }

    pub fn run(&self, x: &Tensor, p: &Prepared, chunk: usize) -> Result<Tensor> {
        match p {
            Prepared::W4A4(raw) => self.apply_w4a4(x, raw, chunk),
            Prepared::Mmq(raw) => self.apply_mmq(x, raw, chunk),
            Prepared::Deq(w) => self.apply(x, w, chunk),
            Prepared::Int8(raw) => self.apply_int8(x, raw, chunk),
            #[cfg(feature = "mlx")]
            Prepared::Mlx(w) => self.apply_mlx(x, w, chunk),
        }
    }

    /// `swiglu(self(x))` with the activation fused into the int8 epilogue when possible.
    pub fn run_swiglu(&self, x: &Tensor, p: &Prepared, chunk: usize) -> Result<Tensor> {
        if self.interleaved {
            if let (Prepared::Int8(raw), true) = (p, self.loras.is_empty()) {
                if let Some(y) = self.try_gemm_s8(x, raw, true)? {
                    return Ok(y);
                }
            }
            return swiglu_interleaved(&self.run(x, p, chunk)?);
        }
        match p {
            Prepared::W4A4(raw) => {
                let y = self.apply_w4a4(x, raw, chunk)?;
                crate::ops::swiglu(&y)
            }
            Prepared::Mmq(raw) => {
                let y = self.apply_mmq(x, raw, chunk)?;
                crate::ops::swiglu(&y)
            }
            Prepared::Int8(raw) if self.loras.is_empty() => {
                let k = x.dim(D::Minus1)?;
                let rows = x.elem_count() / k;
                let x2 = x.reshape((rows, k))?;
                let dev = x.device();
                let fuse = matches!(self.fmt, QFormat::Int8 { convrot: Some(g) } | QFormat::AsymW4A8 { convrot: Some(g), .. } | QFormat::SymW4A4 { convrot: Some(g), .. } | QFormat::AmpereW4A8Preswizzled { convrot: Some(g), .. } if kernels::can_fuse_rotation(k, g));
                let xr = crate::dit::prof::time("  rotate", dev, || match self.fmt {
                    QFormat::Int8 { convrot: Some(g) } | QFormat::AsymW4A8 { convrot: Some(g), .. } | QFormat::AmpereW4A8Preswizzled { convrot: Some(g), .. } if !fuse => rotate_activation(&x2, g),
                    _ => Ok(x2),
                })?;
                let scale = self.scale.as_ref().unwrap().get(x.device())?;
                let _ = chunk;
                crate::dit::prof::time("  int8 gemm", dev, || Ok(kernels::int8_linear_ex(&xr, &raw.w, &scale, self.bias.as_ref(), x.dtype(), true, fuse)?))
            }
            _ => crate::ops::swiglu(&self.run(x, p, chunk)?),
        }
    }

    /// The fused int8 GEMM (sm_80+, bf16 activations, K % 64 == 0), `None` when it doesn't apply.
    fn try_gemm_s8(&self, x: &Tensor, raw: &Raw, swiglu: bool) -> Result<Option<Tensor>> {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let on = *ON.get_or_init(|| std::env::var("H3_GEMM").map(|v| v != "cublaslt").unwrap_or(true));
        let k = x.dim(D::Minus1)?;
        if !on || x.dtype() != DType::BF16 || k % 64 != 0 || self.out_features % 128 != 0 || !crate::attn8::available_arch(x.device()) {
            return Ok(None);
        }
        let rows = x.elem_count() / k;
        let x2 = x.reshape((rows, k))?;
        // the warp-level rotation is fused into the quantizer (group 256, K <= 56 groups)
        let fused_rot = matches!(self.fmt, QFormat::Int8 { convrot: Some(256) } | QFormat::AsymW4A8 { convrot: Some(256), .. } | QFormat::SymW4A4 { convrot: Some(256), .. } | QFormat::AmpereW4A8Preswizzled { convrot: Some(256), .. }) && k / 256 <= 56 && std::env::var("H3_ROT").map(|v| v != "gemm").unwrap_or(true);
        let xr = match self.fmt {
            QFormat::Int8 { convrot: Some(g) } | QFormat::AsymW4A8 { convrot: Some(g), .. } | QFormat::SymW4A4 { convrot: Some(g), .. } | QFormat::AmpereW4A8Preswizzled { convrot: Some(g), .. } if !fused_rot => crate::dit::prof::time("  rotate", x.device(), || rotate_activation(&x2, g))?,
            _ => x2,
        };
        let scale = self.scale.as_ref().unwrap().get(x.device())?;
        let y = crate::dit::prof::time("  int8 gemm", x.device(), || Ok::<_, anyhow::Error>(crate::attn8::gemm_s8(&xr, &raw.w, &scale, self.bias.as_ref(), swiglu, fused_rot)?))?;
        let mut od = x.dims().to_vec();
        *od.last_mut().unwrap() = if swiglu { self.out_features / 2 } else { self.out_features };
        Ok(Some(y.reshape(od)?))
    }

    
    pub fn apply_mmq(&self, x: &Tensor, raw: &Raw, chunk: usize) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let k = *dims.last().unwrap();
        let rows = x.elem_count() / k;
        let x2 = x.reshape((rows, k))?;

        let chunk = if chunk == 0 { 1024 } else { chunk.min(1024) };
        let dev = x.device();
        let _is_q5k = matches!(self.fmt, QFormat::GgufQuant { gtype: GgmlType::Q5K });

        let mut outs = Vec::new();
        let mut start = 0;
        while start < rows {
            let n = chunk.min(rows - start);
            let xc = x2.narrow(0, start, n)?;
            let mut yc = match self.fmt {
                QFormat::GgufQuant { gtype: GgmlType::Q5K } => {
                    crate::dit::prof::time("  mmq_q5k gemm", dev, || {
                        kernels::mmq_q5_k(&xc, &raw.w, self.out_features, self.in_features)
                    })?
                }
                QFormat::GgufQuant { gtype: GgmlType::Q4K } => {
                    crate::dit::prof::time("  mmq_q4k gemm", dev, || {
                        kernels::mmq_q4_k(&xc, &raw.w, self.out_features, self.in_features)
                    })?
                }
                QFormat::GgufQuant { gtype: GgmlType::IQ4XS } => {
                    crate::dit::prof::time("  mmq_iq4xs gemm", dev, || {
                        kernels::mmq_iq4_xs(&xc, &raw.w, self.out_features, self.in_features)
                    })?
                }
                QFormat::GgufQuant { gtype: GgmlType::Q2K } => {
                    crate::dit::prof::time("  mmq_q2k gemm", dev, || {
                        kernels::mmq_q2_k(&xc, &raw.w, self.out_features, self.in_features)
                    })?
                }
                QFormat::GgufQuant { gtype: GgmlType::Q4_0 } => {
                    crate::dit::prof::time("  mmq_q40 gemm", dev, || {
                        kernels::mmq_q4_0(&xc, &raw.w, self.out_features, self.in_features)
                    })?
                }
                QFormat::GgufQuant { gtype: GgmlType::Q6K } => {
                    crate::dit::prof::time("  mmq_q6k gemm", dev, || {
                        kernels::mmq_q6_k(&xc, &raw.w, self.out_features, self.in_features)
                    })?
                }
                QFormat::GgufQuant { gtype: GgmlType::Q8_0 } => {
                    crate::dit::prof::time("  mmq_q80 gemm", dev, || {
                        kernels::mmq_q8_0(&xc, &raw.w, self.out_features, self.in_features)
                    })?
                }
                _ => {
                    crate::dit::prof::time("  mmq_q3k gemm", dev, || {
                        kernels::mmq_q3_k(&xc, &raw.w, self.out_features, self.in_features)
                    })?
                }
            };
            for l in &self.loras {
                let d = xc.matmul(&l.a.to_dtype(x.dtype())?.t()?)?.matmul(&l.b.to_dtype(x.dtype())?.t()?)?;
                yc = (yc + (d * l.scale)?)?;
            }
            outs.push(yc);
            start += n;
        }

        let y = if outs.len() == 1 { outs.pop().unwrap() } else { Tensor::cat(&outs, 0)? };

        let y = match &self.bias {
            Some(b) => y.broadcast_add(&b.to_dtype(x.dtype())?)?,
            None => y,
        };

        let mut od = dims;
        *od.last_mut().unwrap() = self.out_features;
        Ok(y.reshape(od)?)
    }

    pub fn apply_w4a4(&self, x: &Tensor, raw: &Raw, chunk: usize) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let k = *dims.last().unwrap();
        let rows = x.elem_count() / k;
        let x2 = x.reshape((rows, k))?;
        let x2 = match &self.pre_quant_scale {
            Some(s) => x2.broadcast_mul(&s.to_dtype(x.dtype())?)?,
            None => x2,
        };
        let dev = x.device();
        let scale = self.scale.as_ref().unwrap().get(dev)?;
        let chunk = chunk.max(1);
        let mut start = 0;
        let mut outs = vec![];
        while start < rows {
            let n = chunk.min(rows - start);
            let xc = x2.narrow(0, start, n)?;
            let (xr, fused) = match self.fmt {
                QFormat::AmpereW4A4Preswizzled { convrot: Some(_), .. } => (xc.clone(), true),
                _ => (xc.clone(), false),
            };
            let mut y = kernels::gemm_w4a4_preswizzled(&xr, &raw.w, &scale, self.bias.as_ref(), self.out_features, fused)?;
            for l in &self.loras {
                let a = l.a.to_dtype(x.dtype())?;
                let b = l.b.to_dtype(x.dtype())?;
                let d = xc.matmul(&a.t()?)?.matmul(&b.t()?)?;
                y = (y + (d * l.scale))?;
            }
            outs.push(y);
            start += n;
        }
        let y = if outs.len() == 1 { outs.pop().unwrap() } else { Tensor::cat(&outs, 0)? };
        let mut od = dims;
        *od.last_mut().unwrap() = self.out_features;
        Ok(y.reshape(od)?)
    }

    fn apply_int8(&self, x: &Tensor, raw: &Raw, chunk: usize) -> Result<Tensor> {
        if self.loras.is_empty() {
            if let Some(y) = self.try_gemm_s8(x, raw, false)? {
                return Ok(y);
            }
        }
        let dims = x.dims().to_vec();
        let k = *dims.last().unwrap();
        let rows = x.elem_count() / k;
        let x2 = x.reshape((rows, k))?;
        let scale = self.scale.as_ref().unwrap().get(x.device())?;
        // the int32 accumulator is rows x out: keep it near 1 GB
        let chunk = chunk.min((1usize << 28) / self.out_features.max(1)).max(1);
        let mut outs = vec![];
        let mut start = 0;
        while start < rows {
            let n = chunk.min(rows - start);
            let xc = x2.narrow(0, start, n)?;
            let dev = x.device();
            let fuse = matches!(self.fmt, QFormat::Int8 { convrot: Some(g) } | QFormat::AsymW4A8 { convrot: Some(g), .. } | QFormat::SymW4A4 { convrot: Some(g), .. } | QFormat::AmpereW4A8Preswizzled { convrot: Some(g), .. } if kernels::can_fuse_rotation(k, g));
            let xr = crate::dit::prof::time("  rotate", dev, || match self.fmt {
                QFormat::Int8 { convrot: Some(g) } | QFormat::AsymW4A8 { convrot: Some(g), .. } | QFormat::AmpereW4A8Preswizzled { convrot: Some(g), .. } if !fuse => rotate_activation(&xc, g),
                _ => Ok(xc.clone()),
            })?;
            let mut y = crate::dit::prof::time("  int8 gemm", dev, || kernels::int8_linear_ex(&xr, &raw.w, &scale, self.bias.as_ref(), x.dtype(), false, fuse))?;
            for l in &self.loras {
                let d = xc.matmul(&l.a.to_dtype(x.dtype())?.t()?)?.matmul(&l.b.to_dtype(x.dtype())?.t()?)?;
                y = (y + (d * l.scale)?)?;
            }
            outs.push(y);
            start += n;
        }
        let y = if outs.len() == 1 { outs.pop().unwrap() } else { Tensor::cat(&outs, 0)? };
        let mut od = dims;
        *od.last_mut().unwrap() = self.out_features;
        Ok(y.reshape(od)?)
    }
}

#[cfg(not(feature = "cuda"))]
impl Marker {
    pub fn record(_: &Device) -> Self { Self }
    pub fn wait(&self) {}
}

#[cfg(feature = "mlx")]
impl QLinear {
    fn apply_mlx(&self, x: &Tensor, w: &crate::apple::Int8Weight, chunk: usize) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let k = self.in_features;
        let rows = x.elem_count() / k;
        let x2 = x.reshape((rows, k))?;
        let x2 = match &self.pre_quant_scale {
            Some(s) => x2.broadcast_mul(&s.to_dtype(x.dtype())?)?,
            None => x2,
        };
        let mut outs = Vec::new();
        for start in (0..rows).step_by(chunk.max(1)) {
            let xc = x2.narrow(0, start, chunk.max(1).min(rows-start))?;
            let xr = match self.fmt {
                QFormat::Int8 { convrot: Some(g) } | QFormat::AsymW4A8 { convrot: Some(g), .. } | QFormat::SymW4A4 { convrot: Some(g), .. } | QFormat::AmpereW4A8Preswizzled { convrot: Some(g), .. } => rotate_activation(&xc, g)?,
                _ => xc.clone(),
            };
            let mut y = w.forward(&xr)?;
            if let Some(b) = &self.bias { y = y.broadcast_add(&b.to_dtype(y.dtype())?)?; }
            for l in &self.loras {
                let d = xc.matmul(&l.a.to_dtype(x.dtype())?.t()?)?.matmul(&l.b.to_dtype(x.dtype())?.t()?)?;
                y = (y + (d * l.scale)?)?;
            }
            outs.push(y);
        }
        let mut shape = dims; *shape.last_mut().unwrap() = self.out_features;
        Ok(Tensor::cat(&outs, 0)?.reshape(shape)?)
    }
}
