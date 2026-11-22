
#[cfg(feature = "flash-attn")]
extern "C" {
    pub fn run_mha(
        q_ptr: *const core::ffi::c_void,
        k_ptr: *const core::ffi::c_void,
        v_ptr: *const core::ffi::c_void,
        o_ptr: *const core::ffi::c_void,
        softmax_lse_ptr: *const core::ffi::c_void,
        alibi_slopes_ptr: *const core::ffi::c_void,
        cu_seqlens_q_ptr: *const i32,
        cu_seqlens_k_ptr: *const i32,
        q_batch_stride: u32,
        k_batch_stride: u32,
        v_batch_stride: u32,
        o_batch_stride: u32,
        alibi_slopes_batch_stride: u32,
        q_row_stride: u32,
        k_row_stride: u32,
        v_row_stride: u32,
        o_row_stride: u32,
        q_head_stride: u32,
        k_head_stride: u32,
        v_head_stride: u32,
        o_head_stride: u32,
        b: u32,
        h: u32,
        h_k: u32,
        d: u32,
        d_rounded: u32,
        softmax_scale: f32,
        seqlen_q: u32,
        seqlen_k: u32,
        seqlen_q_rounded: u32,
        seqlen_k_rounded: u32,
        total_q: u32,
        is_bf16: core::ffi::c_int,
        is_causal: core::ffi::c_int,
        unpadded_lse: core::ffi::c_int,
        window_size_left: core::ffi::c_int,
        window_size_right: core::ffi::c_int,
        softcap: f32,
        block_table_ptr: *const i32,
        block_table_batch_stride: u32,
        page_block_size: core::ffi::c_int,
        mm_prefix_ranges_ptr: *const i32,
        mm_prefix_range_batch_stride: u32,
        max_mm_prefix_ranges: core::ffi::c_int,
        stream_ptr: *mut core::ffi::c_void,
    );
}

#[cfg(feature = "flash-attn")]
fn round_multiple(x: usize, m: usize) -> usize {
    ((x + m - 1) / m) * m
}

#[cfg(feature = "flash-attn")]
struct FlashAttnWithLseOp {
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
    lse_tensor: Tensor,
}

#[cfg(feature = "flash-attn")]
use candle_core::backend::BackendStorage;

#[cfg(feature = "flash-attn")]
impl FlashAttnWithLseOp {
    fn cuda_fwd_t<
        T: candle_core::cuda_backend::CudaDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &candle_core::CudaStorage,
        q_l: &candle_core::Layout,
        k: &candle_core::CudaStorage,
        k_l: &candle_core::Layout,
        v: &candle_core::CudaStorage,
        v_l: &candle_core::Layout,
        is_bf16: bool,
    ) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};

        let dev = q.device();
        let (b_sz, seqlen_q, num_heads, head_size) = q_l.shape().dims4()?;
        let (_, seqlen_k, num_heads_k, _) = k_l.shape().dims4()?;
        let head_size_rounded = round_multiple(head_size, 32);
        let seqlen_q_rounded = round_multiple(seqlen_q, 128);
        let seqlen_k_rounded = round_multiple(seqlen_k, 128);

        let out_shape = candle_core::Shape::from((b_sz, seqlen_q, num_heads, head_size));
        let mut dst = unsafe { dev.alloc::<T>(out_shape.elem_count())? };

        let q_stride = q_l.stride();
        let k_stride = k_l.stride();
        let v_stride = v_l.stride();
        let o_stride = [seqlen_q * num_heads * head_size, num_heads * head_size, head_size, 1];

        let w_left = self.window_size_left.map(|v| v as i32).unwrap_or(-1);
        let w_right = self.window_size_right.map(|v| v as i32).unwrap_or(-1);

        let stream = dev.cuda_stream();
        let q_slice = q.as_cuda_slice::<T>()?.slice(q_l.start_offset()..);
        let k_slice = k.as_cuda_slice::<T>()?.slice(k_l.start_offset()..);
        let v_slice = v.as_cuda_slice::<T>()?.slice(v_l.start_offset()..);

        let (lse_storage, lse_layout) = self.lse_tensor.storage_and_layout();
        let lse_cuda = match &*lse_storage {
            candle_core::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle_core::bail!("lse must be cuda"),
        };
        let lse_slice = lse_cuda.slice(lse_layout.start_offset()..);

        unsafe {
            let (q_ptr, _gq) = q_slice.device_ptr(&stream);
            let (k_ptr, _gk) = k_slice.device_ptr(&stream);
            let (v_ptr, _gv) = v_slice.device_ptr(&stream);
            let (dst_ptr, _gd) = dst.device_ptr_mut(&stream);
            let (lse_ptr, _gl) = lse_slice.device_ptr(&stream);

            run_mha(
                q_ptr as *const core::ffi::c_void,
                k_ptr as *const core::ffi::c_void,
                v_ptr as *const core::ffi::c_void,
                dst_ptr as *const core::ffi::c_void,
                lse_ptr as *const core::ffi::c_void,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                q_stride[0] as u32,
                k_stride[0] as u32,
                v_stride[0] as u32,
                o_stride[0] as u32,
                0,
                q_stride[1] as u32,
                k_stride[1] as u32,
                v_stride[1] as u32,
                o_stride[1] as u32,
                q_stride[2] as u32,
                k_stride[2] as u32,
                v_stride[2] as u32,
                o_stride[2] as u32,
                b_sz as u32,
                num_heads as u32,
                num_heads_k as u32,
                head_size as u32,
                head_size_rounded as u32,
                self.softmax_scale,
                seqlen_q as u32,
                seqlen_k as u32,
                seqlen_q_rounded as u32,
                seqlen_k_rounded as u32,
                (b_sz * seqlen_q) as u32,
                if is_bf16 { 1 } else { 0 },
                0, // is_causal
                0, // unpadded_lse
                w_left,
                w_right,
                0.0, // softcap
                std::ptr::null(),
                0,
                -1,
                std::ptr::null(),
                0,
                0,
                stream.cu_stream() as *mut core::ffi::c_void,
            );
        }

        let dst = candle_core::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}

#[cfg(feature = "flash-attn")]
impl candle_core::CustomOp3 for FlashAttnWithLseOp {
    fn name(&self) -> &'static str { "flash-attn-lse" }
    fn cpu_fwd(&self, _: &candle_core::CpuStorage, _: &candle_core::Layout, _: &candle_core::CpuStorage, _: &candle_core::Layout, _: &candle_core::CpuStorage, _: &candle_core::Layout) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("cpu not supported")
    }
    fn cuda_fwd(&self, q: &candle_core::CudaStorage, q_l: &candle_core::Layout, k: &candle_core::CudaStorage, k_l: &candle_core::Layout, v: &candle_core::CudaStorage, v_l: &candle_core::Layout) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        match q.dtype() {
            candle_core::DType::F16 => self.cuda_fwd_t::<half::f16>(q, q_l, k, k_l, v, v_l, false),
            candle_core::DType::BF16 => self.cuda_fwd_t::<half::bf16>(q, q_l, k, k_l, v, v_l, true),
            dt => candle_core::bail!("flash-attn-lse is only supported for f16/bf16 ({dt:?})"),
        }
    }
}

#[cfg(feature = "flash-attn")]
pub fn flash_attn_lse(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    w_left: Option<usize>,
    w_right: Option<usize>,
) -> Result<(Tensor, Tensor)> {
    let (b, sq, h, _) = q.dims4()?;
    let lse = Tensor::zeros((b, h, sq), DType::F32, q.device())?;
    let op = FlashAttnWithLseOp {
        softmax_scale: scale,
        window_size_left: w_left,
        window_size_right: w_right,
        lse_tensor: lse.clone(),
    };
    let out = q.apply_op3(k, v, op)?;
    Ok((out, lse))
}

#[cfg(feature = "flash-attn")]
pub fn attention_def107(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    p_global: usize,
    window: usize,
) -> Result<Tensor> {
    let s = q.dim(1)?;
    if s <= p_global {
        return Ok(candle_flash_attn::flash_attn(q, k, v, scale, false)?);
    }

    // 1. Anchor Q attends to ALL K, V densely
    let q_anchor = q.narrow(1, 0, p_global)?.contiguous()?;
    let out_anchor = candle_flash_attn::flash_attn(&q_anchor, k, v, scale, false)?;

    // 2. Video Q attends to Global Anchor (Pass 1) + Local Window (Pass 2)
    let q_video = q.narrow(1, p_global, s - p_global)?.contiguous()?;
    let k_anchor = k.narrow(1, 0, p_global)?.contiguous()?;
    let v_anchor = v.narrow(1, 0, p_global)?.contiguous()?;

    let k_video = k.narrow(1, p_global, s - p_global)?.contiguous()?;
    let v_video = v.narrow(1, p_global, s - p_global)?.contiguous()?;

    // Pass 1: Global Anchor Cross-Attention
    let (out_g, lse_g) = flash_attn_lse(&q_video, &k_anchor, &v_anchor, scale, None, None)?;

    // Pass 2: Local Window Self-Attention
    let (out_w, lse_w) = flash_attn_lse(&q_video, &k_video, &v_video, scale, Some(window), Some(window))?;

    // Online softmax fusion in FP32/F16
    let lse_g = lse_g.transpose(1, 2)?.unsqueeze(3)?;
    let lse_w = lse_w.transpose(1, 2)?.unsqueeze(3)?;

    let m = lse_g.maximum(&lse_w)?;
    let w_g = (&lse_g - &m)?.exp()?.to_dtype(q.dtype())?;
    let w_w = (&lse_w - &m)?.exp()?.to_dtype(q.dtype())?;
    let w_sum = (&w_g + &w_w)?;

    let num_g = out_g.broadcast_mul(&w_g)?;
    let num_w = out_w.broadcast_mul(&w_w)?;
    let num = (&num_g + &num_w)?;
    let out_video = num.broadcast_div(&w_sum)?;

    Ok(Tensor::cat(&[&out_anchor, &out_video], 1)?)
}

/// Shared tensor ops: norms, attention, activations.

use anyhow::Result;
use candle_core::{DType, Tensor, D};

/// RMSNorm over the last dim, weight cast to the input dtype (fused candle kernel).
pub fn rms_norm(x: &Tensor, w: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    let x = x.contiguous()?;
    let dim = x.dim(D::Minus1)?;
    let w = match w {
        Some(w) => w.to_dtype(x.dtype())?,
        None => Tensor::ones(dim, x.dtype(), x.device())?,
    };
    Ok(candle_nn::ops::rms_norm(&x, &w, eps)?)
}

/// LayerNorm over the last dim (computed in f32, returned in the input dtype).
pub fn layer_norm(x: &Tensor, w: Option<&Tensor>, b: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    let dt = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let mut y = xc.broadcast_div(&(var + eps as f64)?.sqrt()?)?;
    if let Some(w) = w {
        y = y.broadcast_mul(&w.to_dtype(DType::F32)?)?;
    }
    if let Some(b) = b {
        y = y.broadcast_add(&b.to_dtype(DType::F32)?)?;
    }
    Ok(y.to_dtype(dt)?)
}

pub fn silu(x: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::silu(x)?)
}

/// `silu(first half) * second half` of the last dim.
pub fn swiglu(x: &Tensor) -> Result<Tensor> {
    let n = x.dim(D::Minus1)? / 2;
    let a = x.narrow(D::Minus1, 0, n)?;
    let b = x.narrow(D::Minus1, n, n)?;
    Ok((candle_nn::ops::silu(&a.contiguous()?)? * b)?)
}

pub fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    Ok(x.gelu()?)
}

pub fn gelu_erf(x: &Tensor) -> Result<Tensor> {
    Ok(x.gelu_erf()?)
}

/// Scaled dot-product attention on `[B, S, H, D]` tensors (k/v may have fewer heads: GQA).
/// Returns `[B, S, H, D]`.
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor> {
    #[cfg(feature = "mlx")]
    if q.device().is_metal() { return crate::apple::attention(q, k, v, causal); }
    let d = q.dim(D::Minus1)?;
    let scale = 1.0 / (d as f32).sqrt();
    #[cfg(feature = "flash-attn")]
    {
        if q.device().is_cuda() && matches!(q.dtype(), DType::F16 | DType::BF16) && d <= 256 {
            let (q, k, v) = (q.contiguous()?, k.contiguous()?, v.contiguous()?);
            // Check DEF-107 Dual-Loop Global Anchor Attention mode:
            if let Ok(def107_val) = std::env::var("H3_DEF107_ANCHOR") {
                let (p_global, window) = if let Some((p_str, w_str)) = def107_val.split_once(',') {
                    (p_str.parse::<usize>().unwrap_or(6801), w_str.parse::<usize>().unwrap_or(8192))
                } else {
                    (6801, def107_val.parse::<usize>().unwrap_or(8192))
                };
                return attention_def107(&q, &k, &v, scale, p_global, window);
            }

            // RC001 Safety Gate: Catch and block unanchored window attempts
            if let Ok(w_str) = std::env::var("H3_ATTN_WINDOW") {
                let allow_unsafe = std::env::var("H3_ALLOW_UNSAFE_WINDOW").map(|v| v == "1").unwrap_or(false);
                if !allow_unsafe {
                    anyhow::bail!(
                        "RC001 VETO: Unanchored window attention (H3_ATTN_WINDOW={}) zeroes out text prompt and Frame-0 anchor! \
                         To run accelerated attention, use DEF-107 Dual-Loop Global Anchor Attention (H3_DEF107_ANCHOR=6801,8192), \
                         or use Full Dense mode (default), or set H3_ALLOW_UNSAFE_WINDOW=1 to explicitly bypass.",
                        w_str
                    );
                }
                if let Ok(w) = w_str.parse::<usize>() {
                    if w > 0 && q.dim(1)? > w {
                        return Ok(candle_flash_attn::flash_attn_windowed(&q, &k, &v, scale, Some(w), Some(w))?);
                    }
                }
            }
            return Ok(candle_flash_attn::flash_attn(&q, &k, &v, scale, causal)?);
        }
    }
    attention_fallback(q, k, v, scale, causal)
}

/// Chunked softmax attention (no flash kernel): processes queries in blocks to bound memory.
pub fn attention_fallback(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, causal: bool) -> Result<Tensor> {
    let (b, s, h, d) = q.dims4()?;
    let hk = k.dim(2)?;
    let rep = h / hk;
    let expand = |t: &Tensor| -> Result<Tensor> {
        let t = t.transpose(1, 2)?.contiguous()?; // [B, Hk, S, D]
        if rep == 1 {
            return Ok(t);
        }
        let (b, hk, s, d) = t.dims4()?;
        Ok(t.unsqueeze(2)?.expand((b, hk, rep, s, d))?.reshape((b, hk * rep, s, d))?)
    };
    let kt = expand(k)?;
    let vt = expand(v)?;
    let sk = kt.dim(2)?;
    let qt = q.transpose(1, 2)?.contiguous()?;
    let chunk = (64 * 1024 * 1024 / (sk * h * 4).max(1)).clamp(1, s);
    let mut outs = vec![];
    let mut start = 0;
    while start < s {
        let n = chunk.min(s - start);
        let qc = qt.narrow(2, start, n)?.to_dtype(DType::F32)?;
        let mut att = (qc.matmul(&kt.to_dtype(DType::F32)?.t()?)? * scale as f64)?;
        if causal {
            let mask: Vec<f32> = (0..n)
                .flat_map(|i| (0..sk).map(move |j| if j > start + i + (sk - s) { f32::NEG_INFINITY } else { 0.0 }))
                .collect();
            let mask = Tensor::from_vec(mask, (n, sk), q.device())?;
            att = att.broadcast_add(&mask)?;
        }
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        outs.push(att.matmul(&vt.to_dtype(DType::F32)?)?.to_dtype(q.dtype())?);
        start += n;
    }
    let o = Tensor::cat(&outs, 2)?; // [B, H, S, D]
    let _ = (b, d);
    Ok(o.transpose(1, 2)?.contiguous()?)
}
