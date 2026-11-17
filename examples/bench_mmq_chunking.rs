use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use h3_rs::kernels;

fn main() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let in_features = 5376usize;
    let out_features = 256usize;
    let m = 4096usize; // 4k tokens

    println!("Benchmarking MMQ chunking on M={}, K={}, N={}", m, in_features, out_features);

    let num_blocks = (out_features * in_features) / 256;
    let raw_bytes_len = num_blocks * 110;
    let mut raw = vec![0x55u8; raw_bytes_len];
    let raw_t = Tensor::from_vec(raw, (raw_bytes_len,), &dev)?;

    let x = Tensor::randn(0.0f32, 1.0, (m, in_features), &dev)?.to_dtype(DType::BF16)?;

    // 1. Unchunked
    dev.synchronize()?;
    let t0 = std::time::Instant::now();
    let _y1 = kernels::mmq_q3_k(&x, &raw_t, out_features, in_features)?;
    dev.synchronize()?;
    let el_unchunked = t0.elapsed();
    println!("Unchunked (M=4096 all at once): {:.3} ms", el_unchunked.as_secs_f64() * 1000.0);

    // 2. Chunked by 128
    for chunk_sz in [64, 128, 256, 512, 1024] {
        dev.synchronize()?;
        let t1 = std::time::Instant::now();
        let mut start = 0;
        let mut outs = Vec::new();
        while start < m {
            let n = chunk_sz.min(m - start);
            let xc = x.narrow(0, start, n)?;
            let yc = kernels::mmq_q3_k(&xc, &raw_t, out_features, in_features)?;
            outs.push(yc);
            start += n;
        }
        let _y_chunked = Tensor::cat(&outs, 0)?;
        dev.synchronize()?;
        let el_chunked = t1.elapsed();
        println!("Chunked by {:4}:                  {:.3} ms", chunk_sz, el_chunked.as_secs_f64() * 1000.0);
    }

    Ok(())
}
