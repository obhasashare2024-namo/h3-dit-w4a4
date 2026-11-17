use anyhow::Result;
use candle_core::Device;
use h3_rs::config::MemoryConfig;
use h3_rs::dit::Dit;
use std::time::Instant;

fn main() -> Result<()> {
    println!("========================================================");
    println!("  MiniMax H3 Comprehensive Fleet Model Loading Test     ");
    println!("========================================================");
    let dev = Device::new_cuda(0)?;

    let test_models = [
        ("FL2VA Q3_K_M GGUF", "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b-Q3_K_M.gguf"),
        ("FL2VA W4A4 Preswizzled ST", "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.safetensors"),
        ("FL2VA W4A4 Preswizzled GGUF", "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.gguf"),
        ("FL2VA W4A8 Preswizzled ST", "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_w4a8_preswizzled.safetensors"),
        ("FL2VA Pruned W4A8 Mixed ST", "/mnt/2tbnvS/models/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors"),
        ("Turbo 4step BF16 ST", "/mnt/2tbnvS/models/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"),
        ("Turbo 8step BF16 ST", "/mnt/2tbnvS/models/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),
        ("Ref2VA Q3_K_M GGUF", "/mnt/R2D/models/minimax_h3_ref2va_hyperflow_flash_14b-Q3_K_M.gguf"),
        ("Ref2VA Pruned W4A8 Mixed ST", "/mnt/R2D/models/minimax_h3_ref2va_pruned_w4a8_mixed.safetensors"),
        ("FL2VA QuantFunc Int4 ST", "/mnt/R2D/models/minimax_h3_fl2va_4step_quantfunc_int4_r128.safetensors"),
        ("FL2VA W4A4 Q4_0 GGUF", "/mnt/R2D/models/minimax_h3_fl2va_w4a4_q4_0.gguf"),
        ("FL2VA W4A8 GGUF", "/mnt/R2D/models/minimax_h3_fl2va_w4a8.gguf"),
        ("FL2VA Pruned Q2_K GGUF", "/mnt/R2D/models/minimax_h3_fl2va_pruned-Q2_K.gguf"),
        ("FL2VA Pruned Q3_K GGUF", "/mnt/R2D/models/minimax_h3_fl2va_pruned-Q3_K.gguf"),
    ];

    let mut mem = MemoryConfig::default();
    mem.dit_vram_blocks = 0;
    mem.dit_ram_blocks = 20;
    mem.dit_disk_blocks = 30;

    let mut passed = 0;
    let mut failed = 0;

    for (name, path) in &test_models {
        print!("Loading {:<30} ... ", name);
        let t0 = Instant::now();
        match Dit::load(path, &dev, &mem, None) {
            Ok(dit) => {
                let dt = t0.elapsed().as_secs_f64();
                let tiers = dit.tiers();
                println!("PASS ({:.2}s, layers={}, tiers=vram:{}/ram:{}/disk:{})", dt, dit.cfg.layers, tiers.vram, tiers.ram, tiers.disk);
                passed += 1;
            }
            Err(e) => {
                println!("FAIL: {e:#}");
                failed += 1;
            }
        }
    }

    println!("========================================================");
    println!("Summary: Passed: {}/{}, Failed: {}", passed, test_models.len(), failed);
    println!("========================================================");
    Ok(())
}
