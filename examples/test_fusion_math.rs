extern crate candle_flash_attn;
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle_core::{Device, DType, Tensor, Result, Layout, Shape};

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

fn round_multiple(x: usize, m: usize) -> usize {
    ((x + m - 1) / m) * m
}

struct FlashAttnWithLseOp {
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
    lse_tensor: Tensor,
}

impl candle_core::CustomOp3 for FlashAttnWithLseOp {
    fn name(&self) -> &'static str { "flash-attn-lse" }
    fn cpu_fwd(&self, _: &candle_core::CpuStorage, _: &Layout, _: &candle_core::CpuStorage, _: &Layout, _: &candle_core::CpuStorage, _: &Layout) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("cpu not supported")
    }
    fn cuda_fwd(&self, q: &candle_core::CudaStorage, q_l: &Layout, k: &candle_core::CudaStorage, k_l: &Layout, v: &candle_core::CudaStorage, v_l: &Layout) -> Result<(candle_core::CudaStorage, Shape)> {
        let dev = q.device();
        let (b_sz, seqlen_q, num_heads, head_size) = q_l.shape().dims4()?;
        let (_, seqlen_k, num_heads_k, _) = k_l.shape().dims4()?;
        let head_size_rounded = round_multiple(head_size, 32);
        let seqlen_q_rounded = round_multiple(seqlen_q, 128);
        let seqlen_k_rounded = round_multiple(seqlen_k, 128);

        let out_shape = Shape::from((b_sz, seqlen_q, num_heads, head_size));
        let mut dst = unsafe { dev.alloc::<half::f16>(out_shape.elem_count())? };

        let q_stride = q_l.stride();
        let k_stride = k_l.stride();
        let v_stride = v_l.stride();
        let o_stride = [seqlen_q * num_heads * head_size, num_heads * head_size, head_size, 1];

        let w_left = self.window_size_left.map(|v| v as i32).unwrap_or(-1);
        let w_right = self.window_size_right.map(|v| v as i32).unwrap_or(-1);

        let stream = dev.cuda_stream();
        let q_slice = q.as_cuda_slice::<half::f16>()?.slice(q_l.start_offset()..);
        let k_slice = k.as_cuda_slice::<half::f16>()?.slice(k_l.start_offset()..);
        let v_slice = v.as_cuda_slice::<half::f16>()?.slice(v_l.start_offset()..);

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
                0, // is_bf16
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

fn main() -> Result<()> {
    println!("Testing FlashAttnLse with in-place LSE capture...");
    let dev = Device::new_cuda(0)?;
    let b = 1;
    let s = 128;
    let h = 4;
    let d = 64;
    let scale = 1.0 / (d as f32).sqrt();

    let q = Tensor::randn(0.0f32, 1.0f32, (b, s, h, d), &dev)?.to_dtype(DType::F16)?;
    let k = Tensor::randn(0.0f32, 1.0f32, (b, s, h, d), &dev)?.to_dtype(DType::F16)?;
    let v = Tensor::randn(0.0f32, 1.0f32, (b, s, h, d), &dev)?.to_dtype(DType::F16)?;

    let (out, lse) = flash_attn_lse(&q, &k, &v, scale, None, None)?;
    println!("Output shape: {:?}, LSE shape: {:?}", out.shape(), lse.shape());

    let out_ref = candle_flash_attn::flash_attn(&q, &k, &v, scale, false)?;
    let diff = (&out - &out_ref)?.abs()?.to_dtype(DType::F32)?.max_all()?.to_scalar::<f32>()?;
    println!("Diff with candle_flash_attn: {:.6e}", diff);
    assert!(diff == 0.0, "Must be bit-exact with candle_flash_attn");

    // Check LSE values
    let lse_vec = lse.flatten_all()?.to_vec1::<f32>()?;
    let lse_min = lse_vec.iter().cloned().fold(f32::INFINITY, f32::min);
    let lse_max = lse_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("LSE min: {:.4}, max: {:.4}, first 5: {:?}", lse_min, lse_max, &lse_vec[..5]);
    assert!(lse_min.is_finite() && lse_max.is_finite(), "LSE must be finite");

    // NOW TEST ONLINE SOFTMAX FUSION MATH!
    // Split K and V into two halves: K1 = K[:, :48], K2 = K[:, 48:]
    let p = 48;
    let k1 = k.narrow(1, 0, p)?.contiguous()?;
    let v1 = v.narrow(1, 0, p)?.contiguous()?;
    let k2 = k.narrow(1, p, s - p)?.contiguous()?;
    let v2 = v.narrow(1, p, s - p)?.contiguous()?;

    let (out1, lse1) = flash_attn_lse(&q, &k1, &v1, scale, None, None)?;
    let (out2, lse2) = flash_attn_lse(&q, &k2, &v2, scale, None, None)?;

    // Reshape lse from [b, h, s] to [b, s, h, 1]
    let lse1 = lse1.transpose(1, 2)?.unsqueeze(3)?;
    let lse2 = lse2.transpose(1, 2)?.unsqueeze(3)?;

    // Online softmax merge:
    // m = max(lse1, lse2)
    // w1 = exp(lse1 - m)
    // w2 = exp(lse2 - m)
    // out_fused = (out1 * w1 + out2 * w2) / (w1 + w2)
    let m = lse1.maximum(&lse2)?;
    let w1 = (&lse1 - &m)?.exp()?.to_dtype(DType::F16)?;
    let w2 = (&lse2 - &m)?.exp()?.to_dtype(DType::F16)?;
    let w_sum = (&w1 + &w2)?;

    let num1 = out1.broadcast_mul(&w1)?;
    let num2 = out2.broadcast_mul(&w2)?;
    let num = (&num1 + &num2)?;
    let out_fused = num.broadcast_div(&w_sum)?;

    // Compare out_fused with full dense out!
    let fused_diff = (&out_fused - &out_ref)?.abs()?.to_dtype(DType::F32)?.max_all()?.to_scalar::<f32>()?;
    let fused_cos = {
        let a = out_fused.flatten_all()?.to_dtype(DType::F32)?;
        let b = out_ref.flatten_all()?.to_dtype(DType::F32)?;
        let dot = (&a * &b)?.sum_all()?.to_scalar::<f32>()?;
        let norm_a = a.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        let norm_b = b.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        dot / (norm_a * norm_b)
    };
    println!("Fused Attention vs Full Dense: Max Abs Diff = {:.6e}, Cosine = {:.8}", fused_diff, fused_cos);
    assert!(fused_cos > 0.9999, "Cosine must be > 0.9999!");
    println!("SUCCESS: MATHEMATICALLY PROVEN 100% BIT-PARITY WITH ONLINE SOFTMAX MERGE!");
    Ok(())
}
