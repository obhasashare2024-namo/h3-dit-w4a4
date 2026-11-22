//! Pinned Host Attention Buffer for long-sequence DiT execution (Phase 3.2b-lite).
//!
//! Allocates a reusable page-locked host memory pool via `cuMemHostAlloc` with Group-Major layout:
//! `[G=28, S, W=256]` in FP16 (2.62 GB for S=182581).
//! Eliminates CPU-side `Tensor::cat` and pageable memory stalling.

use candle_core::cuda_backend::cudarc::driver::sys;
use candle_core::{Result, Tensor};
use std::sync::{Arc, Mutex};

pub struct PinnedAttBuffer {
    ptr: *mut u8,
    total_bytes: usize,
    pub s: usize,
    pub groups: usize,
    pub w: usize,
}

unsafe impl Send for PinnedAttBuffer {}
unsafe impl Sync for PinnedAttBuffer {}

impl PinnedAttBuffer {
    pub fn new(s: usize, groups: usize, w: usize) -> Result<Self> {
        let total_bytes = groups * s * w * 2; // FP16 = 2 bytes
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let res = sys::cuMemHostAlloc(&mut ptr, total_bytes, 0x01); // CU_MEMHOSTALLOC_PORTABLE
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                candle_core::bail!("cuMemHostAlloc failed with error code {:?}", res);
            }
        }
        println!("[PinnedAttBuffer] Allocated {:.2} GB pinned host memory (S={}, G={}, W={})",
                 total_bytes as f64 / 1e9, s, groups, w);
        Ok(Self {
            ptr: ptr as *mut u8,
            total_bytes,
            s,
            groups,
            w,
        })
    }

    pub fn host_ptr(&self, g: usize) -> *mut u8 {
        let bytes_per_group = self.s * self.w * 2;
        unsafe { self.ptr.add(g * bytes_per_group) }
    }

    pub fn host_ptr_offset(&self, g: usize, st: usize) -> *const u8 {
        let offset = (g * self.s + st) * self.w * 2;
        unsafe { self.ptr.add(offset) as *const u8 }
    }

    /// Asynchronously DMA transfer a single attention group output [S, W] from GPU to Host
    pub fn write_group_async(&self, g: usize, o: &Tensor) -> Result<()> {
        let dst_host = self.host_ptr(g);
        crate::kernels::copy_tensor_to_host_pinned_async(o, dst_host)?;
        Ok(())
    }

    /// Asynchronously DMA transfer a single attention group output [S, W] from GPU to Host on a dedicated CUDA stream
    pub fn write_group_async_stream(&self, g: usize, o: &Tensor, stream: sys::CUstream) -> Result<()> {
        let dst_host = self.host_ptr(g);
        crate::kernels::copy_tensor_to_host_pinned_async_stream(o, dst_host, stream)?;
        Ok(())
    }

    /// Asynchronously fetch 28 panels for row chunk [st..st+n] from Host to GPU staging,
    /// then execute gather kernel to produce [n, groups * w] continuous GPU tensor.
    pub fn fetch_chunk_gathered(&self, st: usize, n: usize, staging: &Tensor, dst_gathered: &Tensor) -> Result<()> {
        let bytes_per_panel = n * self.w * 2;
        for g in 0..self.groups {
            let host_src = self.host_ptr_offset(g, st);
            let dst_byte_offset = g * bytes_per_panel;
            crate::kernels::copy_host_pinned_to_tensor_offset_async(host_src, staging, dst_byte_offset, bytes_per_panel)?;
        }
        crate::kernels::gather_groups_to_rows_inplace(dst_gathered, staging, self.groups, self.w)?;
        Ok(())
    }

    /// Asynchronously fetch row chunk [st..st+n] from Host to GPU staging on a dedicated CUDA stream
    pub fn fetch_chunk_staging_stream(&self, st: usize, n: usize, staging: &Tensor, stream: sys::CUstream) -> Result<()> {
        let bytes_per_panel = n * self.w * 2;
        for g in 0..self.groups {
            let host_src = self.host_ptr_offset(g, st);
            let dst_byte_offset = g * bytes_per_panel;
            crate::kernels::copy_host_pinned_to_tensor_offset_async_stream(host_src, staging, dst_byte_offset, bytes_per_panel, stream)?;
        }
        Ok(())
    }
}

impl Drop for PinnedAttBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                sys::cuMemFreeHost(self.ptr as *mut std::ffi::c_void);
            }
        }
    }
}

static PINNED_ATT: Mutex<Option<Arc<PinnedAttBuffer>>> = Mutex::new(None);

pub fn get_or_create_pinned_att(s: usize, groups: usize, w: usize) -> Result<Arc<PinnedAttBuffer>> {
    let mut g = PINNED_ATT.lock().unwrap();
    if let Some(ref buf) = *g {
        if buf.s == s && buf.groups == groups && buf.w == w {
            return Ok(buf.clone());
        }
    }
    let buf = Arc::new(PinnedAttBuffer::new(s, groups, w)?);
    *g = Some(buf.clone());
    Ok(buf)
}
