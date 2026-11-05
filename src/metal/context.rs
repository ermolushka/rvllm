// Metal device, kernel pipelines and the buffers that live on it.
//
// Differences from the CUDA runtime that shape everything below:
//  * Apple GPUs share memory with the CPU. Every buffer is created with
//    `StorageModeShared`, so uploading is a memcpy into `contents()` and
//    reading results back is just a read - no cudaMemcpy equivalents.
//  * Work is recorded, not launched: kernels are encoded into one command
//    buffer / compute encoder and nothing runs until `synchronize` commits it.
//    (CUDA's stream does that queueing implicitly.) A forward pass is one
//    command buffer.
//  * Kernels are compiled from source at startup (`new_library_with_source`),
//    the analogue of NVRTC, so no offline `metal` compiler is needed.
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::marker::PhantomData;

use metal::objc::{msg_send, sel, sel_impl};
use metal::{
    Buffer, BufferRef, CommandBuffer, CommandQueue, CompileOptions, ComputeCommandEncoder,
    ComputePipelineState, Device, MTLCommandBufferStatus, MTLLanguageVersion, MTLResourceOptions,
    MTLSize,
};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const PRELUDE: &str = include_str!("kernels/prelude.metal");
const SOURCES: [&str; 4] = [
    include_str!("kernels/elementwise.metal"),
    include_str!("kernels/rows.metal"),
    include_str!("kernels/attention.metal"),
    include_str!("kernels/matmul.metal"),
];

// Every kernel function in SOURCES; one pipeline state is built for each.
const KERNELS: &[&str] = &[
    "add",
    "silu_glu",
    "gelu_glu",
    "scale_inplace",
    "embed_lookup",
    "narrow_last_row",
    "copy_f32",
    "short_conv_bx",
    "short_conv_filter_gate",
    "rmsnorm",
    "add_rmsnorm",
    "argmax",
    "attn_prep",
    "linear_small",
    "linear_mma_small",
    "linear_gemm_32x64",
];

// `attn_decode` specializations: (query heads per threadgroup, ceil(head_dim / 32), name).
// A group of G heads sharing a KV head is handled by one threadgroup when a
// variant for G exists; otherwise the G = 1 variant runs one head at a time.
pub const ATTN_DECODE_KERNELS: &[(usize, usize, &str)] = &[
    (1, 2, "attn_decode_g1_d2"),
    (2, 2, "attn_decode_g2_d2"),
    (3, 2, "attn_decode_g3_d2"),
    (4, 2, "attn_decode_g4_d2"),
    (6, 2, "attn_decode_g6_d2"),
    (7, 2, "attn_decode_g7_d2"),
    (8, 2, "attn_decode_g8_d2"),
    (1, 4, "attn_decode_g1_d4"),
    (2, 4, "attn_decode_g2_d4"),
    (3, 4, "attn_decode_g3_d4"),
    (4, 4, "attn_decode_g4_d4"),
    (6, 4, "attn_decode_g6_d4"),
    (7, 4, "attn_decode_g7_d4"),
    (1, 8, "attn_decode_g1_d8"),
    (2, 8, "attn_decode_g2_d8"),
    (3, 8, "attn_decode_g3_d8"),
];

// One kernel argument, bound to consecutive buffer indices in order.
pub enum Arg<'a> {
    // A buffer at a byte offset into it (offset 0 for the whole buffer).
    Buf(&'a BufferRef, u64),
    U32(u32),
    F32(f32),
}

// A typed, shared-memory buffer: the `CudaSlice<T>` of this backend.
pub struct MetalBuf<T> {
    buf: Buffer,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: Copy> MetalBuf<T> {
    fn bytes(len: usize) -> u64 {
        // Metal rejects zero-length buffers.
        (len * std::mem::size_of::<T>()).max(4) as u64
    }

    // Zero-filled (Metal zero-initialises new buffers).
    pub fn zeros(rt: &MetalRuntime, len: usize) -> Self {
        let buf = rt
            .device
            .new_buffer(Self::bytes(len), MTLResourceOptions::StorageModeShared);
        MetalBuf {
            buf,
            len,
            _marker: PhantomData,
        }
    }

    // Copy of host data. Only call this for data the GPU is not still writing.
    pub fn from_slice(rt: &MetalRuntime, data: &[T]) -> Self {
        if data.is_empty() {
            return Self::zeros(rt, 0);
        }
        let buf = rt.device.new_buffer_with_data(
            data.as_ptr().cast::<c_void>(),
            Self::bytes(data.len()),
            MTLResourceOptions::StorageModeShared,
        );
        MetalBuf {
            buf,
            len: data.len(),
            _marker: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn arg(&self) -> Arg<'_> {
        Arg::Buf(&self.buf, 0)
    }

    // The buffer bound `elems` elements in, for working on a sub-range.
    pub fn arg_at(&self, elems: usize) -> Arg<'_> {
        Arg::Buf(&self.buf, (elems * std::mem::size_of::<T>()) as u64)
    }
}

struct Pending {
    cmd: CommandBuffer,
    enc: ComputeCommandEncoder,
}

pub struct MetalRuntime {
    pub device: Device,
    queue: CommandQueue,
    pipelines: HashMap<&'static str, ComputePipelineState>,
    // The command buffer being recorded, opened lazily by the first dispatch.
    pending: RefCell<Option<Pending>>,
    // RVLLM_METAL_PROFILE=1: run every dispatch in its own command buffer and
    // total its pure GPU time per kernel (printed when the runtime drops). Much
    // slower than normal runs; for finding where a step's time goes.
    profile: Option<RefCell<HashMap<String, (u64, f64)>>>,
}

impl MetalRuntime {
    pub fn new() -> Result<Self> {
        let device = Device::system_default().ok_or("no Metal device found")?;
        let queue = device.new_command_queue();

        let mut source = String::from(PRELUDE);
        for s in SOURCES {
            source.push('\n');
            source.push_str(s);
        }
        let options = CompileOptions::new();
        // Fast-math lets the compiler assume no inf/NaN. The attention mask is
        // made of -inf and the argmax kernel is specified in terms of NaN, so
        // both need strict IEEE behaviour.
        options.set_fast_math_enabled(false);
        options.set_language_version(MTLLanguageVersion::V3_0);
        let library = device
            .new_library_with_source(&source, &options)
            .map_err(|e| format!("Metal kernel compile failed:\n{e}"))?;

        let mut pipelines = HashMap::new();
        for &name in KERNELS.iter().chain(ATTN_DECODE_KERNELS.iter().map(|(_, _, n)| n)) {
            if pipelines.contains_key(name) {
                continue;
            }
            let function = library
                .get_function(name, None)
                .map_err(|e| format!("kernel {name:?} not found: {e}"))?;
            let pso = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| format!("pipeline for {name:?} failed: {e}"))?;
            pipelines.insert(name, pso);
        }
        Ok(MetalRuntime {
            device,
            queue,
            pipelines,
            pending: RefCell::new(None),
            profile: std::env::var_os("RVLLM_METAL_PROFILE").map(|_| RefCell::new(HashMap::new())),
        })
    }

    // Records one dispatch of `kernel`. `grid` is the total thread count (the
    // hardware handles a ragged last threadgroup), `group` the threadgroup size.
    pub fn dispatch(&self, kernel: &str, args: &[Arg], grid: MTLSize, group: MTLSize) -> Result<()> {
        self.record(kernel, args, |enc| enc.dispatch_threads(grid, group))
    }

    // Like `dispatch`, but the caller picks the number of threadgroups (for
    // kernels that map a threadgroup to a row or a tile).
    pub fn dispatch_groups(
        &self,
        kernel: &str,
        args: &[Arg],
        groups: MTLSize,
        group: MTLSize,
    ) -> Result<()> {
        self.record(kernel, args, |enc| enc.dispatch_thread_groups(groups, group))
    }

    // One thread per element, 1-D.
    pub fn dispatch_1d(&self, kernel: &str, args: &[Arg], n: usize) -> Result<()> {
        let width = 256.min(self.max_threads(kernel)?);
        self.dispatch(
            kernel,
            args,
            MTLSize::new(n as u64, 1, 1),
            MTLSize::new(width, 1, 1),
        )
    }

    fn max_threads(&self, kernel: &str) -> Result<u64> {
        Ok(self
            .pipelines
            .get(kernel)
            .ok_or_else(|| format!("kernel {kernel:?} not found"))?
            .max_total_threads_per_threadgroup())
    }

    // (objc's msg_send! expands to a `cfg(cargo-clippy)` check rustc does not know.)
    #[allow(unexpected_cfgs)]
    fn record(
        &self,
        kernel: &str,
        args: &[Arg],
        encode: impl FnOnce(&ComputeCommandEncoder),
    ) -> Result<()> {
        let pso = self
            .pipelines
            .get(kernel)
            .ok_or_else(|| format!("kernel {kernel:?} not found"))?;
        if let Some(profile) = &self.profile {
            let cmd = self.queue.new_command_buffer().to_owned();
            let enc = cmd.new_compute_command_encoder().to_owned();
            Self::bind(&enc, pso, args);
            encode(&enc);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            let (start, end): (f64, f64) = unsafe { (msg_send![&*cmd, GPUStartTime], msg_send![&*cmd, GPUEndTime]) };
            // Matmuls are keyed by shape: the same kernel serves very different jobs.
            let key = match (kernel.starts_with("linear_"), args) {
                (true, [.., Arg::U32(rows), Arg::U32(in_f), Arg::U32(out_f)]) => {
                    format!("{kernel} rows={rows} {in_f}x{out_f}")
                }
                // Attention: prefill (q_len > 1) and decode cost very different amounts.
                _ if kernel.starts_with("attn_decode") => match args {
                    [.., Arg::U32(q_len), Arg::F32(_)] => format!("attn_decode q_len={}", (*q_len).min(2)),
                    _ => kernel.to_string(),
                },
                _ => kernel.to_string(),
            };
            let mut totals = profile.borrow_mut();
            let entry = totals.entry(key).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += end - start;
            return Ok(());
        }
        let mut pending = self.pending.borrow_mut();
        let pending = pending.get_or_insert_with(|| {
            let cmd = self.queue.new_command_buffer().to_owned();
            let enc = cmd.new_compute_command_encoder().to_owned();
            Pending { cmd, enc }
        });
        Self::bind(&pending.enc, pso, args);
        encode(&pending.enc);
        Ok(())
    }

    fn bind(enc: &ComputeCommandEncoder, pso: &ComputePipelineState, args: &[Arg]) {
        enc.set_compute_pipeline_state(pso);
        for (i, arg) in args.iter().enumerate() {
            let i = i as u64;
            match arg {
                Arg::Buf(buf, offset) => enc.set_buffer(i, Some(buf), *offset),
                Arg::U32(v) => enc.set_bytes(i, 4, (v as *const u32).cast()),
                Arg::F32(v) => enc.set_bytes(i, 4, (v as *const f32).cast()),
            }
        }
    }

    // Commits everything recorded so far and waits for the GPU to finish it.
    pub fn synchronize(&self) -> Result<()> {
        let Some(pending) = self.pending.borrow_mut().take() else {
            return Ok(());
        };
        pending.enc.end_encoding();
        pending.cmd.commit();
        pending.cmd.wait_until_completed();
        if pending.cmd.status() == MTLCommandBufferStatus::Error {
            return Err("Metal command buffer failed".into());
        }
        Ok(())
    }

    // Host copy of a buffer's contents, after the GPU has finished with it.
    pub fn read<T: Copy>(&self, buf: &MetalBuf<T>) -> Result<Vec<T>> {
        self.synchronize()?;
        let mut out = Vec::with_capacity(buf.len);
        unsafe {
            std::ptr::copy_nonoverlapping(buf.buf.contents().cast::<T>(), out.as_mut_ptr(), buf.len);
            out.set_len(buf.len);
        }
        Ok(out)
    }

    // A new buffer holding a copy of `src`, made on the GPU in dispatch order.
    pub fn clone_buf(&self, src: &MetalBuf<f32>) -> Result<MetalBuf<f32>> {
        let out = MetalBuf::<f32>::zeros(self, src.len());
        self.copy_f32(src, 0, &out, 0, src.len())?;
        Ok(out)
    }

    // dst[dst_off..dst_off + n] = src[src_off..src_off + n], in dispatch order.
    pub fn copy_f32(
        &self,
        src: &MetalBuf<f32>,
        src_off: usize,
        dst: &MetalBuf<f32>,
        dst_off: usize,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        if src_off + n > src.len() || dst_off + n > dst.len() {
            return Err("copy_f32: range out of bounds".into());
        }
        self.dispatch_1d(
            "copy_f32",
            &[src.arg_at(src_off), dst.arg_at(dst_off), Arg::U32(n as u32)],
            n,
        )
    }
}

impl Drop for MetalRuntime {
    // An encoder dropped without `end_encoding` is a hard Metal assertion, which
    // would hide the real error when a panic unwinds through a half-recorded pass.
    fn drop(&mut self) {
        if let Some(pending) = self.pending.get_mut().take() {
            pending.enc.end_encoding();
        }
        if let Some(profile) = &self.profile {
            let profile = profile.borrow();
            let total: f64 = profile.values().map(|v| v.1).sum();
            let mut rows: Vec<_> = profile.iter().collect();
            rows.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
            eprintln!("metal gpu time by kernel (total {:.1} ms):", total * 1e3);
            for (key, (count, secs)) in rows.into_iter().take(40) {
                eprintln!(
                    "  {key:<44} {count:>7} calls {:>9.1} ms {:>7.1} us/call {:>5.1}%",
                    secs * 1e3,
                    secs * 1e6 / *count as f64,
                    secs / total * 100.0
                );
            }
        }
    }
}

// Runs `f` inside an autorelease pool. Metal hands back command buffers and
// encoders as autoreleased objects; on a thread without a pool (every thread
// but the main one) they, and the buffers they keep alive, would leak.
pub fn with_pool<R>(f: impl FnOnce() -> R) -> R {
    metal::objc::rc::autoreleasepool(f)
}

// For GPU tests: a runtime on the default device, or None when the machine has
// no Metal device at all (the test is then skipped with a note on stderr). A
// device that exists but fails to initialise - most likely a kernel that does
// not compile - is a real failure, never a skip. Set RVLLM_REQUIRE_METAL to
// make a missing device a failure too.
#[cfg(test)]
pub fn test_runtime() -> Option<MetalRuntime> {
    if Device::system_default().is_none() {
        assert!(
            std::env::var_os("RVLLM_REQUIRE_METAL").is_none(),
            "RVLLM_REQUIRE_METAL is set but there is no Metal device"
        );
        eprintln!("skipping Metal test: no Metal device");
        return None;
    }
    Some(MetalRuntime::new().expect("Metal runtime init failed"))
}
