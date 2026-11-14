code
Rust
download
content_copy
expand_less
// ============================================================================
// Part 2: Rust Implementation (src/w4a4_ampere.rs)
// NVRTC Compilation, PTX Module Loading, Device Memory Management,
// Kernel Launch Wrappers (via cudarc), and FFI Bindings.
// ============================================================================

use std::sync::Arc;
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig, DriverError,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

pub const W4A4_AMPERE_CUDA_SRC: &str = include_str!("w4a4_ampere_kernels.cu");

const MODULE_NAME: &str = "w4a4_ampere_module";
const QUANT_KERNEL_NAME: &str = "w4a4_h256_dynamic_quant_kernel";
const GEMM_KERNEL_NAME: &str = "w4a4_ampere_gemm_swizzle_pipeline_kernel";

#[derive(Debug)]
pub enum W4A4Error {
    Driver(DriverError),
    Nvrtc(cudarc::nvrtc::CompileError),
    InvalidDimension(&'static str),
}

impl From<DriverError> for W4A4Error {
    fn from(e: DriverError) -> Self {
        W4A4Error::Driver(e)
    }
}

impl From<cudarc::nvrtc::CompileError> for W4A4Error {
    fn from(e: cudarc::nvrtc::CompileError) -> Self {
        W4A4Error::Nvrtc(e)
    }
}

pub struct QuantizedTensorW4 {
    pub packed_s4: CudaSlice<u32>,
    pub scales: CudaSlice<f32>,
    pub rows: usize,
    pub cols: usize,
}

pub struct W4A4AmpereEngine {
    device: Arc<CudaDevice>,
    quant_fn: CudaFunction,
    gemm_fn: CudaFunction,
}

impl W4A4AmpereEngine {
    pub fn new(device: Arc<CudaDevice>, sm_arch: &'static str) -> Result<Self, W4A4Error> {
        let opts = CompileOptions {
            arch: Some(sm_arch), // e.g., "compute_80" or "sm_80"
            use_fast_math: Some(true),
            options: vec![
                "--std=c++17".to_string(),
                "-default-device".to_string(),
            ],
            ..Default::default()
        };

        let ptx = compile_ptx_with_opts(W4A4_AMPERE_CUDA_SRC, opts)?;
        device.load_ptx(
            ptx,
            MODULE_NAME,
            &[QUANT_KERNEL_NAME, GEMM_KERNEL_NAME],
        )?;

        let quant_fn = device
            .get_func(MODULE_NAME, QUANT_KERNEL_NAME)
            .expect("Failed to load w4a4_h256_dynamic_quant_kernel");
        let gemm_fn = device
            .get_func(MODULE_NAME, GEMM_KERNEL_NAME)
            .expect("Failed to load w4a4_ampere_gemm_swizzle_pipeline_kernel");

        Ok(Self {
            device,
            quant_fn,
            gemm_fn,
        })
    }

    pub fn quantize_h256_into(
        &self,
        input_f16: &CudaSlice<u16>,
        output_packed_s4: &mut CudaSlice<u32>,
        output_scales: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<(), W4A4Error> {
        if cols == 0 || (cols & 255) != 0 {
            return Err(W4A4Error::InvalidDimension(
                "K dimension must be a non-zero multiple of 256",
            ));
        }

        let num_groups = rows * (cols >> 8);
        let threads_per_block: u32 = 256; // 8 warps per CTA
        let warps_per_block: u32 = threads_per_block >> 5;
        let grid_x = ((num_groups as u32) + warps_per_block - 1) / warps_per_block;

        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.quant_fn.clone().launch(
                cfg,
                (
                    input_f16,
                    output_packed_s4,
                    output_scales,
                    num_groups as i32,
                ),
            )?;
        }

        Ok(())
    }

    pub fn quantize_h256(
        &self,
        input_f16: &CudaSlice<u16>,
        rows: usize,
        cols: usize,
    ) -> Result<QuantizedTensorW4, W4A4Error> {
        if cols == 0 || (cols & 255) != 0 {
            return Err(W4A4Error::InvalidDimension(
                "K dimension must be a non-zero multiple of 256",
            ));
        }

        let num_groups = rows * (cols >> 8);
        let mut packed_s4 = unsafe { self.device.alloc::<u32>(num_groups * 32)? };
        let mut scales = unsafe { self.device.alloc::<f32>(num_groups)? };

        self.quantize_h256_into(input_f16, &mut packed_s4, &mut scales, rows, cols)?;

        Ok(QuantizedTensorW4 {
            packed_s4,
            scales,
            rows,
            cols,
        })
    }

    pub fn gemm_w4a4_into(
        &self,
        a: &QuantizedTensorW4,
        b: &QuantizedTensorW4,
        c_f16: &mut CudaSlice<u16>,
    ) -> Result<(), W4A4Error> {
        if a.cols != b.cols || (a.cols & 255) != 0 {
            return Err(W4A4Error::InvalidDimension(
                "A and B K dimensions must match and be a multiple of 256",
            ));
        }

        let m = a.rows;
        let n = b.rows;
        let k = a.cols;

        let grid_x = ((n as u32) + 63) / 64;
        let grid_y = ((m as u32) + 63) / 64;

        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0, // Uses 32 KB static shared memory
        };

        unsafe {
            self.gemm_fn.clone().launch(
                cfg,
                (
                    &a.packed_s4,
                    &b.packed_s4,
                    &a.scales,
                    &b.scales,
                    c_f16,
                    m as i32,
                    n as i32,
                    k as i32,
                ),
            )?;
        }

        Ok(())
    }

    pub fn forward_w4a4_linear(
        &self,
        x_f16: &CudaSlice<u16>,
        w_quant: &QuantizedTensorW4,
        m: usize,
    ) -> Result<CudaSlice<u16>, W4A4Error> {
        let k = w_quant.cols;
        let n = w_quant.rows;

        let x_quant = self.quantize_h256(x_f16, m, k)?;
        let mut out_f16 = unsafe { self.device.alloc::<u16>(m * n)? };
        self.gemm_w4a4_into(&x_quant, w_quant, &mut out_f16)?;

        Ok(out_f16)
    }
}

// ============================================================================
// Optional C-FFI Bindings (When Linked via NVCC / build.rs)
// ============================================================================
pub mod ffi {
    use std::ffi::c_void;

    pub type CudaError_t = i32;
    pub type CudaStream_t = *mut c_void;

    extern "C" {
        pub fn launch_w4a4_h256_dynamic_quant(
            input_f16: *const c_void,
            output_packed_s4: *mut c_void,
            output_scales: *mut f32,
            num_groups: i32,
            stream: CudaStream_t,
        ) -> CudaError_t;

        pub fn launch_w4a4_ampere_gemm_swizzle_pipeline(
            a_packed: *const c_void,
            b_packed: *const c_void,
            scale_a: *const f32,
            scale_b: *const f32,
            c_f16: *mut c_void,
            m: i32,
            n: i32,
            k: i32,
            stream: CudaStream_t,
        ) -> CudaError_t;
    }
}
