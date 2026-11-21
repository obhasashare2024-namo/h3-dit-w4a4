// File: src/streaming.rs
// 3-Tier Zero-Allocation Streaming Engine:
// - Task 1: Static Double-Buffered VRAM StagingRing<2>
// - Task 2: Pure CUDA Event Stream-to-Stream Synchronization (Zero Host Stalls)
// - Task 3: Contiguous Write-Combined Pinned RAM & Async DiskPrefetchRing<3>
// ============================================================================

use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};

use anyhow::{bail, Context, Result};
use candle_core::cuda_backend::cudarc::driver::{sys, DevicePtr};
use candle_core::cuda_backend::cudarc::driver::sys::CUevent_flags_enum::CU_EVENT_DISABLE_TIMING;
use candle_core::cuda_backend::cudarc::driver::sys::CUstream_flags_enum::CU_STREAM_NON_BLOCKING;
use candle_core::cuda_backend::CudaStorageSlice;
use candle_core::{DType, Device, Storage, Tensor};

use crate::weights::{Raw, SafeFile};

pub const NUM_VRAM_STAGING_SLOTS: usize = 2;
pub const NUM_DISK_RING_SLOTS: usize = 3;

#[inline]
fn check_cu(res: sys::CUresult, msg: &'static str) -> Result<()> {
    if res != sys::CUresult::CUDA_SUCCESS {
        bail!("CUDA Driver Error in {msg}: {res:?}");
    }
    Ok(())
}

/// Metadata for the 4 projections [qkv, out, fc1, fc2] packed contiguously inside a 192.68 MB block buffer.
#[derive(Debug, Clone)]
pub struct PackedProjMeta {
    pub file_offset: u64,
    pub block_offset: usize,
    pub byte_len: usize,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct BlockPackLayout {
    pub projs: [PackedProjMeta; 4],
    pub total_bytes: usize,
}

impl BlockPackLayout {
    pub fn from_file(file: &SafeFile, block_idx: usize) -> Result<Self> {
        let names = [
            format!("blocks.{block_idx}.attn.qkv_proj.weight"),
            format!("blocks.{block_idx}.attn.out_proj.weight"),
            format!("blocks.{block_idx}.mlp.fc1.weight"),
            format!("blocks.{block_idx}.mlp.fc2.weight"),
        ];
        let mut cursor = 0usize;
        let mut projs = Vec::with_capacity(4);
        for name in &names {
            let info = file.info(name)?;
            let byte_len = info.end - info.start;
            let file_offset = (file.data_start() + info.start) as u64;
            let block_offset = cursor;
            // Align each projection to 256 bytes for coalesced 16B/128B Tensor Core access
            cursor = (cursor + byte_len + 255) & !255;
            projs.push(PackedProjMeta {
                file_offset,
                block_offset,
                byte_len,
                shape: info.shape.clone(),
            });
        }
        Ok(Self {
            projs: projs.try_into().unwrap(),
            total_bytes: cursor,
        })
    }
}

/// Page-locked host buffer allocated via `cuMemHostAlloc` (or `mmap` + `cuMemHostRegister_v2` fallback).
/// Guarantees 100% pinned residency for maximum PCIe DMA bandwidth (~12.8 GB/s PCIe 3.0 / ~25.2 GB/s PCIe 4.0).
pub struct ContiguousPinnedBuffer {
    ptr: *mut u8,
    len: usize,
    is_registered_mmap: bool,
}

unsafe impl Send for ContiguousPinnedBuffer {}
unsafe impl Sync for ContiguousPinnedBuffer {}

impl ContiguousPinnedBuffer {
    pub fn alloc(len: usize, write_combined: bool) -> Result<Self> {
        let alloc_len = len.max(4096);
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let flags = if write_combined {
            sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_WRITECOMBINED
        } else {
            sys::CU_MEMHOSTALLOC_PORTABLE
        };
        let res = unsafe { sys::cuMemHostAlloc(&mut ptr, alloc_len, flags) };
        if res == sys::CUresult::CUDA_SUCCESS && !ptr.is_null() {
            return Ok(Self {
                ptr: ptr as *mut u8,
                len: alloc_len,
                is_registered_mmap: false,
            });
        }

        // Fallback if cuMemHostAlloc hits fragmentation: anonymous page-aligned allocation + cuMemHostRegister
        let layout = std::alloc::Layout::from_size_align(alloc_len, 4096)?;
        let raw_ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if raw_ptr.is_null() {
            bail!("Host RAM allocation failed for {} bytes", alloc_len);
        }
        let reg_res = unsafe {
            sys::cuMemHostRegister_v2(
                raw_ptr as *mut std::ffi::c_void,
                alloc_len,
                sys::CU_MEMHOSTREGISTER_PORTABLE,
            )
        };
        check_cu(reg_res, "cuMemHostRegister_v2")?;
        Ok(Self {
            ptr: raw_ptr,
            len: alloc_len,
            is_registered_mmap: true,
        })
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn as_mut_slice(&mut self, len: usize) -> &mut [u8] {
        assert!(len <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr, len) }
    }
}

impl Drop for ContiguousPinnedBuffer {
    fn drop(&mut self) {
        unsafe {
            if self.is_registered_mmap {
                let _ = sys::cuMemHostUnregister(self.ptr as *mut std::ffi::c_void);
                if let Ok(layout) = std::alloc::Layout::from_size_align(self.len, 4096) {
                    std::alloc::dealloc(self.ptr, layout);
                }
            } else {
                let _ = sys::cuMemFreeHost(self.ptr as *mut std::ffi::c_void);
            }
        }
    }
}

/// Where an offloaded block's weights live.
pub enum OffloadedBlockSource {
    /// Resident in a single contiguous Write-Combined Pinned RAM buffer (14 blocks).
    PinnedRam {
        buf: Arc<ContiguousPinnedBuffer>,
        layout: BlockPackLayout,
    },
    /// Resident on NVMe Disk, pre-staged into `DiskPrefetchRing<3>` by the background IO thread (5 blocks).
    Disk {
        disk_ord: usize, // 0..num_disk_blocks
        layout: BlockPackLayout,
    },
}

// ============================================================================
// Task 3B: Asynchronous Disk IO Ring Buffer (DiskPrefetchRing<3>)
// ============================================================================

unsafe impl Send for DiskSlotState {}
unsafe impl Sync for DiskSlotState {}

struct DiskSlotState {
    buf: ContiguousPinnedBuffer,
    ev_htod_done: sys::CUevent,
    has_recorded: AtomicBool,
    ready_tag: Mutex<u64>,
    cvar: Condvar,
}

enum DiskWorkerCmd {
    /// Prefetch all disk blocks for sampling step `step_id`.
    PrefetchStep { step_id: u64 },
    Shutdown,
}

pub struct DiskPrefetchRing {
    slots: Arc<Vec<DiskSlotState>>,
    tx: mpsc::Sender<DiskWorkerCmd>,
    step_counter: AtomicU64,
}

impl DiskPrefetchRing {
    pub fn new(
        file: Arc<SafeFile>,
        disk_layouts: Vec<BlockPackLayout>,
        max_block_bytes: usize,
        cu_ctx: sys::CUcontext,
    ) -> Result<Self> {
        let num_slots = disk_layouts.len();
        let mut slots_vec = Vec::with_capacity(num_slots);
        for _ in 0..num_slots {
            let mut ev: sys::CUevent = std::ptr::null_mut();
            check_cu(
                unsafe { sys::cuEventCreate(&mut ev, CU_EVENT_DISABLE_TIMING as u32) },
                "cuEventCreate(disk_slot)",
            )?;
            slots_vec.push(DiskSlotState {
                buf: ContiguousPinnedBuffer::alloc(max_block_bytes, false)?,
                ev_htod_done: ev,
                has_recorded: AtomicBool::new(false),
                ready_tag: Mutex::new(u64::MAX),
                cvar: Condvar::new(),
            });
        }
        let slots = Arc::new(slots_vec);

        let (tx, rx) = mpsc::channel::<DiskWorkerCmd>();
        let slots_bg = slots.clone();
        let ctx_addr = cu_ctx as usize;

        std::thread::Builder::new()
            .name("h3-disk-prefetch".into())
            .spawn(move || {
                unsafe {
                    let _ = sys::cuCtxSetCurrent(ctx_addr as sys::CUcontext);
                }
                while let Ok(cmd) = rx.recv() {
                    match cmd {
                        DiskWorkerCmd::PrefetchStep { step_id } => {
                            for (disk_ord, layout) in disk_layouts.iter().enumerate() {
                                let slot = &slots_bg[disk_ord];

                                // Wait only on the background IO thread until the previous DMA from this
                                // pinned slot into VRAM has completed on stream_copy.
                                if slot.has_recorded.load(Ordering::Acquire) {
                                    unsafe {
                                        let _ = sys::cuEventSynchronize(slot.ev_htod_done);
                                    }
                                }

                                // Perform large sequential pread directly into the pinned host buffer
                                let dst_ptr = slot.buf.as_ptr() as *mut u8;
                                for proj in &layout.projs {
                                    let dst_slice = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            dst_ptr.add(proj.block_offset),
                                            proj.byte_len,
                                        )
                                    };
                                    let _ = file
                                        .file_handle()
                                        .read_exact_at(dst_slice, proj.file_offset);
                                }

                                // Signal that (step_id, disk_ord) is ready in pinned RAM
                                let tag = (step_id << 16) | (disk_ord as u64);
                                let mut g = slot.ready_tag.lock().unwrap();
                                *g = tag;
                                slot.cvar.notify_all();
                            }
                        }
                        DiskWorkerCmd::Shutdown => break,
                    }
                }
            })?;

        Ok(Self {
            slots,
            tx,
            step_counter: AtomicU64::new(1),
        })
    }

    /// Kick off asynchronous `pread` for all Disk-tier blocks at the start of an Euler step.
    pub fn begin_step(&self) -> u64 {
        let step_id = self.step_counter.fetch_add(1, Ordering::Relaxed);
        let _ = self.tx.send(DiskWorkerCmd::PrefetchStep { step_id });
        step_id
    }

    /// Wait until Disk block `disk_ord` for `step_id` has been read into its pinned slot,
    /// then return `(pinned_host_ptr, ring_slot_idx)`.
    pub fn acquire_pinned_slot(&self, step_id: u64, disk_ord: usize) -> (*const u8, usize) {
        let slot = &self.slots[disk_ord];
        let want_tag = (step_id << 16) | (disk_ord as u64);
        let mut g = slot.ready_tag.lock().unwrap();
        while *g != want_tag {
            g = slot.cvar.wait(g).unwrap();
        }
        (slot.buf.as_ptr(), disk_ord)
    }

    /// Record on `stream_copy` that the HtoD DMA from `ring_idx` has been enqueued.
    pub fn record_htod_done(&self, ring_idx: usize, stream_copy: sys::CUstream) -> Result<()> {
        let slot = &self.slots[ring_idx];
        check_cu(
            unsafe { sys::cuEventRecord(slot.ev_htod_done, stream_copy) },
            "cuEventRecord(disk_htod_done)",
        )?;
        slot.has_recorded.store(true, Ordering::Release);
        Ok(())
    }
}

impl Drop for DiskPrefetchRing {
    fn drop(&mut self) {
        let _ = self.tx.send(DiskWorkerCmd::Shutdown);
        for s in self.slots.iter() {
            unsafe {
                let _ = sys::cuEventDestroy_v2(s.ev_htod_done);
            }
        }
    }
}

// ============================================================================
// Task 1 & Task 2: Static Double-Buffered VRAM StagingRing<2> & Event Ping-Pong
// ============================================================================

struct VramSlot {
    /// Persistent U8 Tensor of shape `[max_block_bytes]`. Never dropped during server lifetime.
    arena: Tensor,
    /// Raw device pointer (`CUdeviceptr`) of `arena`.
    dptr: u64,
    /// Recorded on `stream_copy` when HtoD DMA into this slot finishes.
    ev_copy_done: sys::CUevent,
    /// Recorded on `stream_compute` when `run_block` finishes consuming this slot.
    ev_compute_done: sys::CUevent,
}

pub struct StreamingEngine {
    stream_compute: sys::CUstream,
    stream_copy: sys::CUstream,
    vram_slots: [VramSlot; NUM_VRAM_STAGING_SLOTS],
    offloaded_blocks: Vec<OffloadedBlockSource>,
    block_indices: Vec<usize>,
    disk_ring: Option<DiskPrefetchRing>,
    _resident_count: usize,
}

unsafe impl Send for StreamingEngine {}
unsafe impl Sync for StreamingEngine {}

/// Extract the raw `CUdeviceptr` from a 1D `DType::U8` Candle `Tensor`.
fn tensor_storage_dptr(t: &Tensor) -> Result<u64> {
    let (st, l) = t.storage_and_layout();
    let Storage::Cuda(c) = &*st else {
        bail!("Expected CUDA tensor for VRAM staging slot");
    };
    let stream = c.device.cuda_stream();
    let (ptr, _guard) = match &c.slice {
        CudaStorageSlice::I32(slice) => slice.device_ptr(&stream),
        CudaStorageSlice::U8(slice) => slice.device_ptr(&stream),
        _ => bail!("Expected I32 or U8 storage slice"),
    };
    Ok(ptr + (l.start_offset() as u64 * t.dtype().size_in_bytes() as u64))
}

impl StreamingEngine {
    pub fn new(
        file: &Arc<SafeFile>,
        dev: &Device,
        total_layers: usize,
        vram_blocks: usize,
        ram_blocks: usize,
        max_block_bytes: usize,
    ) -> Result<Self> {
        let Device::Cuda(cd) = dev else {
            bail!("StreamingEngine requires a CUDA device");
        };

        let stream_compute = cd.cuda_stream().cu_stream();
        let cu_ctx = cd.cuda_stream().context().cu_ctx();

        // Create dedicated non-blocking HtoD copy stream
        let mut stream_copy: sys::CUstream = std::ptr::null_mut();
        check_cu(
            unsafe { sys::cuStreamCreate(&mut stream_copy, CU_STREAM_NON_BLOCKING as u32) },
            "cuStreamCreate(stream_copy)",
        )?;

        // Allocate the 2 static VRAM staging buffers (Buffer A & Buffer B) once at startup
        let alloc_slot = |idx: usize| -> Result<VramSlot> {
            let max_words = max_block_bytes.div_ceil(4);
            let arena = Tensor::zeros(max_words, DType::I32, dev)
                .with_context(|| format!("Allocating VRAM StagingRing slot {idx}"))?;
            let dptr = tensor_storage_dptr(&arena)?;
            let mut ev_copy_done: sys::CUevent = std::ptr::null_mut();
            let mut ev_compute_done: sys::CUevent = std::ptr::null_mut();
            unsafe {
                check_cu(
                    sys::cuEventCreate(&mut ev_copy_done, CU_EVENT_DISABLE_TIMING as u32),
                    "cuEventCreate(ev_copy_done)",
                )?;
                check_cu(
                    sys::cuEventCreate(&mut ev_compute_done, CU_EVENT_DISABLE_TIMING as u32),
                    "cuEventCreate(ev_compute_done)",
                )?;
                // Record initial compute_done event so the first cuStreamWaitEvent is immediately satisfied
                check_cu(
                    sys::cuEventRecord(ev_compute_done, stream_compute),
                    "cuEventRecord(init ev_compute_done)",
                )?;
            }
            Ok(VramSlot {
                arena,
                dptr,
                ev_copy_done,
                ev_compute_done,
            })
        };

        let vram_slots = [alloc_slot(0)?, alloc_slot(1)?];

        // Materialize the 14 RAM blocks into contiguous Write-Combined Pinned RAM buffers
        // and register the 5 Disk blocks into the asynchronous DiskPrefetchRing<3>.
        let mut offloaded_blocks = Vec::with_capacity(total_layers.saturating_sub(vram_blocks));
        let mut block_indices = Vec::with_capacity(total_layers.saturating_sub(vram_blocks));
        let mut disk_layouts = Vec::new();

        for bi in vram_blocks..total_layers {
            let is_fused = file.has(&format!("blocks.{bi}.attn.qkv_proj.weight"))
                && file.has(&format!("blocks.{bi}.attn.out_proj.weight"))
                && file.has(&format!("blocks.{bi}.mlp.fc1.weight"))
                && file.has(&format!("blocks.{bi}.mlp.fc2.weight"));
            if !is_fused {
                continue;
            }
            block_indices.push(bi);
            let layout = BlockPackLayout::from_file(file, bi)?;
            if bi < vram_blocks + ram_blocks {
                let mut pinned = ContiguousPinnedBuffer::alloc(layout.total_bytes, true)?;
                let dst = pinned.as_mut_slice(layout.total_bytes);
                for proj in &layout.projs {
                    file. file_handle().read_exact_at(
                        &mut dst[proj.block_offset..proj.block_offset + proj.byte_len],
                        proj.file_offset,
                    )?;
                }
                offloaded_blocks.push(OffloadedBlockSource::PinnedRam {
                    buf: Arc::new(pinned),
                    layout,
                });
            } else {
                let disk_ord = disk_layouts.len();
                disk_layouts.push(layout.clone());
                offloaded_blocks.push(OffloadedBlockSource::Disk { disk_ord, layout });
            }
        }

        let disk_ring = if !disk_layouts.is_empty() {
            Some(DiskPrefetchRing::new(
                file.clone(),
                disk_layouts,
                max_block_bytes,
                cu_ctx,
            )?)
        } else {
            None
        };

        Ok(Self {
            stream_compute,
            stream_copy,
            vram_slots,
            offloaded_blocks,
            disk_ring,
            block_indices,
            _resident_count: vram_blocks,
        })
    }

    #[inline]
    pub fn offload_index_of(&self, bi: usize) -> Option<usize> {
        self.block_indices.iter().position(|&b| b == bi)
    }

    pub fn has_offloaded(&self) -> bool {
        !self.offloaded_blocks.is_empty()
    }

    /// Start background Disk `pread` for the current Euler step and immediately prefetch
    /// the first two offloaded blocks (`off_idx = 0` into Slot 0, `off_idx = 1` into Slot 1)
    /// while the GPU is busy computing the 28 VRAM-resident blocks!
    pub fn begin_step_and_prime(&self) -> Result<u64> {
        let step_id = self
            .disk_ring
            .as_ref()
            .map(|dr| dr.begin_step())
            .unwrap_or(0);

        if !self.offloaded_blocks.is_empty() {
            self.enqueue_htod_async(0, step_id)?;
        }
        if self.offloaded_blocks.len() > 1 {
            self.enqueue_htod_async(1, step_id)?;
        }
        Ok(step_id)
    }

    /// Enqueue an asynchronous 192.68 MB HtoD DMA transfer for `off_idx = bi - resident_count`
    /// into `slot_idx = off_idx & 1` on `stream_copy`.
    /// Never blocks or synchronizes the host CPU thread on GPU execution.
    pub fn enqueue_htod_async(&self, off_idx: usize, step_id: u64) -> Result<()> {
        if off_idx >= self.offloaded_blocks.len() {
            return Ok(());
        }
        let slot_idx = off_idx & 1;
        let slot = &self.vram_slots[slot_idx];

        // 1. GPU-side wait: stream_copy waits until stream_compute finishes using `slot`
        check_cu(
            unsafe { sys::cuStreamWaitEvent(self.stream_copy, slot.ev_compute_done, 0) },
            "cuStreamWaitEvent(stream_copy <- ev_compute_done)",
        )?;

        // 2. Single coalesced 192.68 MB async DMA transfer on stream_copy
        match &self.offloaded_blocks[off_idx] {
            OffloadedBlockSource::PinnedRam { buf, layout } => {
                check_cu(
                    unsafe {
                        sys::cuMemcpyHtoDAsync_v2(
                            slot.dptr,
                            buf.as_ptr() as *const std::ffi::c_void,
                            layout.total_bytes,
                            self.stream_copy,
                        )
                    },
                    "cuMemcpyHtoDAsync_v2(PinnedRam -> VRAM Slot)",
                )?;
            }
            OffloadedBlockSource::Disk { disk_ord, layout } => {
                let dr = self.disk_ring.as_ref().expect("disk_ring initialized");
                let (host_ptr, disk_ring_slot) = dr.acquire_pinned_slot(step_id, *disk_ord);
                check_cu(
                    unsafe {
                        sys::cuMemcpyHtoDAsync_v2(
                            slot.dptr,
                            host_ptr as *const std::ffi::c_void,
                            layout.total_bytes,
                            self.stream_copy,
                        )
                    },
                    "cuMemcpyHtoDAsync_v2(DiskPinnedSlot -> VRAM Slot)",
                )?;
                dr.record_htod_done(disk_ring_slot, self.stream_copy)?;
            }
        }

        // 3. Record copy completion event on stream_copy
        check_cu(
            unsafe { sys::cuEventRecord(slot.ev_copy_done, self.stream_copy) },
            "cuEventRecord(ev_copy_done)",
        )?;

        Ok(())
    }

    /// Wait on `stream_compute` (GPU-side) for `slot = off_idx & 1` HtoD copy to complete,
    /// and return 4 zero-allocation `Raw` tensor views `[qkv, out, fc1, fc2]` into the slot arena.
    pub fn acquire_compute_slot(&self, off_idx: usize) -> Result<[Raw; 4]> {
        let slot_idx = off_idx & 1;
        let slot = &self.vram_slots[slot_idx];

        // GPU-side wait: stream_compute waits until stream_copy finishes uploading `off_idx`
        check_cu(
            unsafe { sys::cuStreamWaitEvent(self.stream_compute, slot.ev_copy_done, 0) },
            "cuStreamWaitEvent(stream_compute <- ev_copy_done)",
        )?;

        let layout = match &self.offloaded_blocks[off_idx] {
            OffloadedBlockSource::PinnedRam { layout, .. } => layout,
            OffloadedBlockSource::Disk { layout, .. } => layout,
        };

        // Construct zero-copy Tensor sub-views backed by `slot.arena`'s persistent Arc<Storage>
        let mut raws = Vec::with_capacity(4);
        for proj in &layout.projs {
            let word_offset = proj.block_offset / 4;
            let word_len = proj.byte_len / 4;
            let view = slot
                .arena
                .narrow(0, word_offset, word_len)?
                .reshape(proj.shape.as_slice())?;
            raws.push(Raw { w: view, bs: None });
        }
        Ok(raws.try_into().ok().unwrap())
    }

    /// Record on `stream_compute` that all kernels for `off_idx` have been enqueued,
    /// and immediately enqueue the HtoD transfer for `off_idx + 2` into the same slot.
    pub fn release_compute_slot_and_prefetch(&self, off_idx: usize, step_id: u64) -> Result<()> {
        let slot_idx = off_idx & 1;
        let slot = &self.vram_slots[slot_idx];

        check_cu(
            unsafe { sys::cuEventRecord(slot.ev_compute_done, self.stream_compute) },
            "cuEventRecord(ev_compute_done)",
        )?;

        // Double-buffer lookahead: slot `off_idx & 1` will next be used by `off_idx + 2`
        if off_idx + 2 < self.offloaded_blocks.len() {
            self.enqueue_htod_async(off_idx + 2, step_id)?;
        }
        Ok(())
    }
}

impl Drop for StreamingEngine {
    fn drop(&mut self) {
        unsafe {
            for s in &self.vram_slots {
                let _ = sys::cuEventDestroy_v2(s.ev_copy_done);
                let _ = sys::cuEventDestroy_v2(s.ev_compute_done);
            }
            let _ = sys::cuStreamDestroy_v2(self.stream_copy);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType, Tensor};

    #[test]
    fn test_narrow_reshape() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let t = Tensor::zeros(1024 * 1024, DType::U8, &dev)?;
        let sub = t.narrow(0, 4096, 1000)?;
        let reshaped = sub.reshape((10, 10, 10))?;
        assert_eq!(reshaped.dims(), &[10, 10, 10]);
        println!("Reshape on 1D narrowed CUDA tensor passed!");
        Ok(())
    }
}
