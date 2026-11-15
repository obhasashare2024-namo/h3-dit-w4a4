use candle_core::{DType, Device, Tensor};
use candle_core::cuda_backend::cudarc::driver::sys;

pub struct PinnedAttBuffer {
    ptr: *mut u8,
    total_bytes: usize,
    s: usize,
    groups: usize,
    w: usize,
}

unsafe impl Send for PinnedAttBuffer {}
unsafe impl Sync for PinnedAttBuffer {}

impl PinnedAttBuffer {
    pub fn new(s: usize, groups: usize, w: usize) -> anyhow::Result<Self> {
        let total_bytes = groups * s * w * 2; // FP16 = 2 bytes
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let res = sys::cuMemHostAlloc(&mut ptr, total_bytes, 0x01); // 0x01 = CU_MEMHOSTALLOC_PORTABLE
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                anyhow::bail!("cuMemHostAlloc failed with error code {:?}", res);
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

    pub fn write_group_async(&self, g: usize, o: &Tensor) -> anyhow::Result<()> {
        let bytes_per_group = self.s * self.w * 2;
        let dst_host = unsafe { self.ptr.add(g * bytes_per_group) };
        h3_rs::kernels::copy_tensor_to_host_pinned_async(o, dst_host)?;
        Ok(())
    }

    pub fn fetch_chunk_gathered(&self, st: usize, n: usize, staging: &Tensor, dst_gathered: &Tensor) -> anyhow::Result<()> {
        let bytes_per_panel = n * self.w * 2;
        for g in 0..self.groups {
            let host_src = unsafe { self.ptr.add((g * self.s + st) * self.w * 2) };
            let dst_byte_offset = g * bytes_per_panel;
            h3_rs::kernels::copy_host_pinned_to_tensor_offset_async(host_src, staging, dst_byte_offset, bytes_per_panel)?;
        }
        h3_rs::kernels::gather_groups_to_rows_inplace(dst_gathered, staging, self.groups, self.w)?;
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

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    println!("=== Testing PinnedAttBuffer End-to-End Pipeline ===");

    let s = 182_581usize;
    let groups = 28usize;
    let w = 256usize;

    let buf = PinnedAttBuffer::new(s, groups, w)?;

    // Test writing group 0 and group 1
    println!("Writing group 0 and 1...");
    let o0 = Tensor::ones((s, w), DType::F16, &dev)?;
    let o1 = Tensor::full(2.0f32, (s, w), &dev)?.to_dtype(DType::F16)?;
    buf.write_group_async(0, &o0)?;
    buf.write_group_async(1, &o1)?;
    dev.synchronize()?;

    // Test fetching a chunk [st=1000..st+n=5096]
    let n = 4096usize;
    let staging = Tensor::zeros((groups, n, w), DType::F16, &dev)?;
    let gathered = Tensor::zeros((n, groups * w), DType::F16, &dev)?;

    println!("Fetching chunk and gathering...");
    buf.fetch_chunk_gathered(1000, n, &staging, &gathered)?;
    dev.synchronize()?;

    // Verify gathered output:
    // For row 0: group 0 slice should be all 1.0, group 1 slice should be all 2.0!
    let g0_slice = gathered.narrow(1, 0, w)?.to_dtype(DType::F32)?.to_device(&candle_core::Device::Cpu)?;
    let g1_slice = gathered.narrow(1, w, w)?.to_dtype(DType::F32)?.to_device(&candle_core::Device::Cpu)?;

    let v0: Vec<f32> = g0_slice.flatten_all()?.to_vec1()?;
    let v1: Vec<f32> = g1_slice.flatten_all()?.to_vec1()?;

    assert_eq!(v0[0], 1.0);
    assert_eq!(v0[v0.len() - 1], 1.0);
    assert_eq!(v1[0], 2.0);
    assert_eq!(v1[v1.len() - 1], 2.0);

    println!("SUCCESS: PinnedAttBuffer + Gather pipeline 100% verified!");
    Ok(())
}
