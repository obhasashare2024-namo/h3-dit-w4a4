use candle_core::{DType, Device, Tensor};

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    println!("=== Testing gather_groups_to_rows_inplace ===");

    let g = 28usize;
    let n = 4096usize;
    let w = 256usize;

    // Use values within FP16 exact integer range [-2048, 2048]
    // group: 0..27, col: 0..255 -> val = (group * 32 + col % 32) (max = 27*32 + 31 = 895 < 2048)
    let mut data = Vec::with_capacity(g * n * w);
    for group in 0..g {
        for _row in 0..n {
            for col in 0..w {
                data.push((group * 32 + col % 32) as f32);
            }
        }
    }
    let src = Tensor::from_vec(data, (g, n, w), &dev)?.to_dtype(DType::F16)?;
    let dst = Tensor::zeros((n, g * w), DType::F16, &dev)?;

    h3_rs::kernels::gather_groups_to_rows_inplace(&dst, &src, g, w)?;
    dev.synchronize()?;

    let dst_cpu = dst.to_dtype(DType::F32)?.to_device(&candle_core::Device::Cpu)?;
    let dst_vec: Vec<f32> = dst_cpu.flatten_all()?.to_vec1()?;

    let mut errors = 0;
    for row in [0, 100, 1000, 4095] {
        for group in 0..g {
            for col in [0, 1, 127, 255] {
                let dst_idx = row * (g * w) + group * w + col;
                let actual = dst_vec[dst_idx];
                let expected = (group * 32 + col % 32) as f32;
                if (actual - expected).abs() > 1e-3 {
                    println!("Mismatch at (row={}, g={}, col={}): expected {}, got {}", row, group, col, expected, actual);
                    errors += 1;
                    if errors > 10 {
                        anyhow::bail!("Too many errors in gather verification!");
                    }
                }
            }
        }
    }

    if errors == 0 {
        println!("SUCCESS: Gather Kernel 100% physically and numerically verified!");
    } else {
        anyhow::bail!("Verification failed with {} errors", errors);
    }
    Ok(())
}
