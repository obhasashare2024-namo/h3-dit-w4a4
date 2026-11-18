use std::fs::File;
use std::io::{Read, Write};
use std::sync::mpsc as std_mpsc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use clap::Parser;
use h3_rs::config::MemoryConfig;
use h3_rs::dit::{Dit, DitCond, KeyframeCond};
use h3_rs::sampling::{sample_denoised, simple_sigmas, Sampler};
use ntex::util::Bytes;
use ntex::web::{self, App, HttpResponse, HttpServer};
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

#[derive(Parser, Debug)]
#[command(name = "server_5a", about = "MiniMax H3 Phase 5A Universal Multi-Quant DiT Microservice on Port 9425")]
struct Args {
    #[arg(long, default_value = "9425")]
    port: u16,

    #[arg(long, default_value = "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b-Q3_K_M.gguf")]
    model: String,

    #[arg(long, default_value_t = -1)]
    blocks: i64,

    #[arg(long, default_value_t = -1)]
    vram_blocks: i64,

    #[arg(long, default_value_t = -1)]
    ram_blocks: i64,

    #[arg(long, default_value_t = -1)]
    disk_blocks: i64,

    #[arg(long)]
    tier_ratio: Option<String>,

    #[arg(long, default_value_t = 1.0)]
    reserve_gb: f64,

    #[arg(long, default_value_t = 4096)]
    token_chunk: usize,

    #[arg(long, default_value_t = 2048)]
    mlp_chunk: usize,
}

#[derive(Deserialize, Debug)]
struct SampleReq {
    latent_path: String,
    #[serde(default = "default_output_path")]
    output_path: String,
    #[serde(default = "default_steps")]
    steps: usize,
    #[serde(default = "default_seed")]
    seed: u64,
    #[serde(default = "default_false")]
    render: bool,
    #[serde(default = "default_render_target")]
    render_target: String,
    #[serde(default = "default_render_video_placement")]
    render_video_placement: String,
    #[serde(default = "default_render_audio_device")]
    render_audio_device: String,
}

fn default_false() -> bool { false }
fn default_output_path() -> String {
    "/mnt/R2D/tmp/minimax_bridge/stage2_denoised_phase5a_9425.edlat".to_string()
}
fn default_steps() -> usize { 6 }
fn default_seed() -> u64 { 42 }

#[derive(Serialize, Debug)]
struct SampleResp {
    status: String,
    model: String,
    out_path: String,
    file_size: u64,
    steps: usize,
    elapsed_seconds: f64,
    seconds_per_step: f64,
    resident_blocks: String,
    tier_distribution: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    rendered_media: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    render_elapsed_seconds: Option<f64>,
}

struct SampleJob {
    req: SampleReq,
    reply: oneshot::Sender<Result<SampleResp>>,
}

fn read_h3lt_tensor<R: Read>(r: &mut R, dev: &Device) -> Result<Tensor> {
    let mut hdr = [0u8; 80];
    r.read_exact(&mut hdr)?;
    let magic = u32::from_le_bytes(hdr[0..4].try_into()?);
    if magic != 0x48334c54 {
        bail!("Invalid H3LT tensor magic: {:#x}", magic);
    }
    let dims = u32::from_le_bytes(hdr[4..8].try_into()?) as usize;
    let mut shape = Vec::with_capacity(dims);
    for i in 0..dims {
        let d = i64::from_le_bytes(hdr[8 + i * 8..16 + i * 8].try_into()?) as usize;
        shape.push(d);
    }
    let numel = u64::from_le_bytes(hdr[72..80].try_into()?) as usize;

    let mut data = vec![0f32; numel];
    let slice_u8 = unsafe {
        std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, numel * 4)
    };
    r.read_exact(slice_u8)?;
    shape.reverse();
    Ok(Tensor::from_vec(data, shape, dev)?)
}

fn write_edlat_tensor(t: &Tensor, out_f: &mut File) -> Result<()> {
    let shape = t.dims();
    let dims: Vec<i64> = shape.iter().map(|&x| x as i64).rev().collect();
    let numel = t.elem_count() as u64;

    let mut shape8 = [0i64; 8];
    for (i, &s) in dims.iter().enumerate().take(8) {
        shape8[i] = s;
    }

    let magic = 0x48334C54u32;
    out_f.write_all(&magic.to_le_bytes())?;
    out_f.write_all(&(dims.len() as u32).to_le_bytes())?;
    for &s in &shape8 {
        out_f.write_all(&s.to_le_bytes())?;
    }
    out_f.write_all(&numel.to_le_bytes())?;
    let data = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    out_f.write_all(bytes)?;
    Ok(())
}

fn worker_loop(
    model_path: String,
    blocks: i64,
    vram_blocks: i64,
    ram_blocks: i64,
    disk_blocks: i64,
    tier_ratio: Option<String>,
    reserve_gb: f64,
    token_chunk: usize,
    mlp_chunk: usize,
    rx: std_mpsc::Receiver<SampleJob>,
) {
    println!("[server_5a] Initializing CUDA Device 0 on worker thread...");
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[server_5a] Fatal: Failed to initialize CUDA: {e}");
            return;
        }
    };
    h3_rs::kernels::keep_pool_memory(&dev);

    println!("[server_5a] Loading W4A4 DiT: {model_path}...");
    let mut mem = MemoryConfig::default();
    mem.dit_gpu_blocks = blocks;
    mem.dit_vram_blocks = if vram_blocks != -1 { vram_blocks } else { blocks };
    mem.dit_ram_blocks = ram_blocks;
    mem.dit_disk_blocks = disk_blocks;
    mem.tier_ratio = tier_ratio;
    mem.reserve_gb = reserve_gb;
    mem.token_chunk = token_chunk;
    mem.mlp_chunk = mlp_chunk;
    mem.offload = h3_rs::weights::Placement::Disk;

    let t0 = Instant::now();
    let dit = match Dit::load(&model_path, &dev, &mem, None) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[server_5a] Fatal: Failed to load DiT: {e}");
            return;
        }
    };
    let tiers = dit.tiers();
    let res_info = format!("vram:{}/ram:{}/disk:{}/total:{}", tiers.vram, tiers.ram, tiers.disk, dit.cfg.layers);
    println!("[server_5a] DiT loaded in {:.2}s, tiers: {res_info}", t0.elapsed().as_secs_f64());

    while let Ok(job) = rx.recv() {
        let req = job.req;
        println!("[server_5a] Handling sample request: latent={}, out={}, steps={}", req.latent_path, req.output_path, req.steps);

        let t_start = Instant::now();
        let res = (|| -> Result<SampleResp> {
            let mut f = File::open(&req.latent_path).context("Open latent file")?;
            let mut magic = [0u8; 8];
            f.read_exact(&mut magic)?;
            if &magic != b"H3STG1\x01\x00" {
                bail!("Invalid H3STG1 bundle magic");
            }

            let mut header = [0i32; 4];
            for h in &mut header {
                let mut b = [0u8; 4];
                f.read_exact(&mut b)?;
                *h = i32::from_le_bytes(b);
            }
            let mut shifts = [0f32; 2];
            for s in &mut shifts {
                let mut b = [0u8; 4];
                f.read_exact(&mut b)?;
                *s = f32::from_le_bytes(b);
            }

            let context = read_h3lt_tensor(&mut f, &dev)?;
            let token_tags_t = read_h3lt_tensor(&mut f, &Device::Cpu)?;
            let vid_lat = read_h3lt_tensor(&mut f, &dev)?;
            let aud_lat = read_h3lt_tensor(&mut f, &dev)?;
            let kf_lat = read_h3lt_tensor(&mut f, &dev)?;

            let mut kf_idx_bytes = [0u8; 4];
            f.read_exact(&mut kf_idx_bytes)?;
            let kf_idx = i32::from_le_bytes(kf_idx_bytes) as usize;

            let token_tags_vec: Vec<u32> = token_tags_t.to_vec1::<f32>()?.iter().map(|&x| x as u32).collect();

            let cond = DitCond {
                text: context,
                tags: Some(token_tags_vec),
                keyframes: vec![KeyframeCond {
                    frame_index: kf_idx,
                    video: Some(kf_lat),
                    audio: None,
                }],
                refs: vec![],
                seed: req.seed,
                visual_cond_noise_aug: 0.0,
                audio_cond_noise_aug: 0.0,
                denoise: None,
                control: None,
            };

            let video_shape = vid_lat.dims();
            let audio_t = aud_lat.dim(3)?;
            let sigmas = simple_sigmas(shifts[0] as f64, req.steps);

            let run = dit.prepare(&cond, video_shape, audio_t, sigmas.clone())?;

            let mut last_t = Instant::now();
            let sampled = sample_denoised(
                &dit,
                &run,
                vid_lat,
                aud_lat,
                &sigmas,
                Sampler::Euler,
                None,
                |step, total| {
                    let step_ms = last_t.elapsed().as_secs_f64() * 1000.0;
                    println!("[server_5a] Step {}/{} done in {:.2}ms", step + 1, total, step_ms);
                    last_t = Instant::now();
                    Ok(())
                },
            )?;

            let edlat_path = if req.output_path.ends_with(".pt") {
                req.output_path.replace(".pt", ".edlat")
            } else {
                req.output_path.clone()
            };

            let mut out_f = File::create(&edlat_path)?;
            write_edlat_tensor(&sampled.denoised_v, &mut out_f)?;
            write_edlat_tensor(&sampled.denoised_a, &mut out_f)?;
            out_f.flush()?;
            let mut f_size = std::fs::metadata(&edlat_path)?.len();

            if req.output_path.ends_with(".pt") {
                let status = std::process::Command::new("python3")
                    .arg("/mnt/R2D/tmp/minimax_bridge/convert_edlat_to_pt.py")
                    .arg(&edlat_path)
                    .arg(&req.output_path)
                    .status();
                if let Ok(st) = status {
                    if st.success() {
                        f_size = std::fs::metadata(&req.output_path)?.len();
                    }
                }
            }

            let (rendered_media, render_elapsed) = if req.render {
                let t_rend = Instant::now();
                let media_out = if req.render_target == "audio" {
                    edlat_path.replace(".edlat", ".wav")
                } else {
                    edlat_path.replace(".edlat", ".mp4")
                };
                let bin_path = "/mnt/2tbnvS/h3-rs/target/release/examples/pipeline_e2e_av";
                let status = std::process::Command::new(bin_path)
                    .arg("--target").arg(&req.render_target)
                    .arg("--input").arg(&edlat_path)
                    .arg("--output").arg(&media_out)
                    .arg("--video-placement").arg(&req.render_video_placement)
                    .arg("--audio-device").arg(&req.render_audio_device)
                    .status();
                if let Ok(st) = status {
                    if st.success() {
                        (Some(media_out), Some((t_rend.elapsed().as_secs_f64() * 100.0).round() / 100.0))
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

            let elapsed = t_start.elapsed().as_secs_f64();
            Ok(SampleResp {
                status: "success".to_string(),
                model: model_path.clone(),
                out_path: req.output_path,
                file_size: f_size,
                steps: req.steps,
                elapsed_seconds: (elapsed * 100.0).round() / 100.0,
                seconds_per_step: ((elapsed / req.steps as f64) * 100.0).round() / 100.0,
                resident_blocks: res_info.clone(),
                tier_distribution: res_info.clone(),
                rendered_media,
                render_elapsed_seconds: render_elapsed,
            })
        })();

        if let Err(ref e) = res {
            eprintln!("[server_5a] Request FAILED with error:
{e:?}");
        }
        let _ = job.reply.send(res);
    }
}


#[derive(Deserialize, Debug)]
struct RenderReq {
    #[serde(default = "default_render_input")]
    input_path: String,
    #[serde(default = "default_render_output")]
    output_path: String,
    #[serde(default = "default_render_target")]
    target: String,
    #[serde(default = "default_render_video_placement")]
    video_placement: String,
    #[serde(default = "default_render_audio_device")]
    audio_device: String,
}

fn default_render_input() -> String {
    "/mnt/R2D/tmp/minimax_bridge/stage2_denoised_w4a4_9412_3s.edlat".to_string()
}
fn default_render_output() -> String {
    "/mnt/2tbnvS/tmp/h3_server_rendered_output.mp4".to_string()
}
fn default_render_target() -> String {
    "both".to_string()
}
fn default_render_video_placement() -> String {
    "ram".to_string()
}
fn default_render_audio_device() -> String {
    "cpu".to_string()
}

async fn handle_render(body: Bytes) -> HttpResponse {
    let req: RenderReq = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return HttpResponse::BadRequest().json(&serde_json::json!({"error": format!("Invalid JSON: {e}")})),
    };
    let t0 = Instant::now();
    let bin_path = "/mnt/2tbnvS/h3-rs/target/release/examples/pipeline_e2e_av";
    let (reply, rx) = oneshot::channel();
    let r_clone = (req.target.clone(), req.input_path.clone(), req.output_path.clone(), req.video_placement.clone(), req.audio_device.clone());
    std::thread::spawn(move || {
        let (target, input, output, v_place, a_dev) = r_clone;
        let out = std::process::Command::new(bin_path)
            .arg("--target").arg(target)
            .arg("--input").arg(input)
            .arg("--output").arg(output)
            .arg("--video-placement").arg(v_place)
            .arg("--audio-device").arg(a_dev)
            .output();
        let _ = reply.send(out);
    });

    match rx.await {
        Ok(Ok(out)) => {
            let elapsed = t0.elapsed().as_secs_f64();
            if out.status.success() {
                let f_size = std::fs::metadata(&req.output_path).map(|m| m.len()).unwrap_or(0);
                HttpResponse::Ok().json(&serde_json::json!({
                    "status": "success",
                    "output_path": req.output_path,
                    "file_size": f_size,
                    "target": req.target,
                    "elapsed_seconds": (elapsed * 100.0).round() / 100.0,
                    "stdout": String::from_utf8_lossy(&out.stdout).to_string(),
                }))
            } else {
                let err_str = String::from_utf8_lossy(&out.stderr).to_string();
                let out_str = String::from_utf8_lossy(&out.stdout).to_string();
                HttpResponse::InternalServerError().json(&serde_json::json!({
                    "status": "error",
                    "stderr": err_str,
                    "stdout": out_str,
                }))
            }
        }
        Ok(Err(e)) => HttpResponse::InternalServerError().json(&serde_json::json!({"error": format!("Command failed: {e}")})),
        Err(_) => HttpResponse::ServiceUnavailable().json(&serde_json::json!({"error": "Render thread canceled"})),
    }
}

async fn handle_sample(tx: std_mpsc::Sender<SampleJob>, body: Bytes) -> HttpResponse {
    let req: SampleReq = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return HttpResponse::BadRequest().json(&serde_json::json!({"error": format!("Invalid JSON: {e}")})),
    };
    let (reply, rx) = oneshot::channel();
    if tx.send(SampleJob { req, reply }).is_err() {
        return HttpResponse::ServiceUnavailable().json(&serde_json::json!({"error": "Worker thread stopped"}));
    }
    match rx.await {
        Ok(Ok(resp)) => HttpResponse::Ok().json(&resp),
        Ok(Err(e)) => HttpResponse::InternalServerError().json(&serde_json::json!({"error": format!("{e:#}")})),
        Err(_) => HttpResponse::ServiceUnavailable().json(&serde_json::json!({"error": "Worker canceled request"})),
    }
}

#[ntex::main(name = "server_5a", signals = true)]
async fn main() -> std::io::Result<()> {
    let args = Args::parse();
    println!("=== Starting MiniMax H3 W4A4 DiT Microservice (Port {}) ===", args.port);

    let (tx, rx) = std_mpsc::channel();
    let model_path = args.model.clone();
    let blocks = args.blocks;
    let vram_blocks = args.vram_blocks;
    let ram_blocks = args.ram_blocks;
    let disk_blocks = args.disk_blocks;
    let tier_ratio = args.tier_ratio.clone();
    let reserve_gb = args.reserve_gb;
    let token_chunk = args.token_chunk;
    let mlp_chunk = args.mlp_chunk;
    std::thread::spawn(move || {
        worker_loop(model_path, blocks, vram_blocks, ram_blocks, disk_blocks, tier_ratio, reserve_gb, token_chunk, mlp_chunk, rx);
    });

    let addr = format!("0.0.0.0:{}", args.port);
    let tx_main = tx.clone();

    HttpServer::new(async move |_| {
        let tx = tx_main.clone();
        App::new()
            .service(web::resource("/health").to(async || {
                HttpResponse::Ok().json(&serde_json::json!({
                    "status": "ok",
                    "service": "fleet-dit-5a",
                    "backend": "pure-rust-nvrtc-phase5a-gguf"
                }))
            }))
            .service(web::resource("/sample").route(web::post().to(move |body: Bytes| handle_sample(tx.clone(), body))))
            .service(web::resource("/render").route(web::post().to(handle_render)))
    })
    .bind(addr.as_str(), ntex::SharedCfg::new("server_5a"))?
    .workers(1)
    .run()
    .await
}
