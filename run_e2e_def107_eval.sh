#!/usr/bin/env bash
set -e

LOG="/mnt/2tbnvS/tmp/run_e2e_v67_1088p_8s_def107_eval.log"
exec > >(tee -a "$LOG") 2>&1

echo "================================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Version 6.7 1088P 8s DEF-107 Dual-Loop Fusion"
echo "Schedule: Step 1 Full Dense (Semantic Lock-in) + Steps 2-6 DEF-107 Fusion (Anchor=6801, Window=8192)"
echo "================================================================================"

# 1. Clean previous runs
pkill -f "server_9411" || true
sleep 2

# 2. Set optimized environment variables
export H3_ATTN="flash"
export H3_ATTN_GROUPS="14"
export H3_STREAMING="1"
export H3_FORCE_GPU_ATT="0"
export H3_ZERO_OFFLOAD="1"
export H3_ATTN_SCHEDULE="def107"
export H3_GLOBAL_ANCHOR_LEN="6801"
export H3_WINDOW_SIZE="8192"

SERVER_LOG="/mnt/2tbnvS/tmp/server_9411_v67_def107.log"
rm -f "$SERVER_LOG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching server_9411 with DEF-107 Fusion..."
/mnt/2tbnvS/h3-rs/target/release/server_9411 \
  --port 9411 \
  --model /mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.safetensors \
  --vram-blocks 0 \
  --ram-blocks 50 \
  --token-chunk 4096 \
  --mlp-chunk 2048 \
  --attn flash \
  > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "server_9411 started with PID $SERVER_PID"

# Wait for server ready (port 9411)
echo "Waiting for server_9411 to be ready on port 9411..."
for i in {1..60}; do
    if curl -s http://127.0.0.1:9411/health >/dev/null 2>&1; then
        echo "server_9411 is READY!"
        break
    fi
    sleep 1
done

# 3. Request 6-step 1088P sampling
IN_EDBIN="/mnt/2tbnvS/tmp/stage1_input_1080p_8s.edbin"
OUT_EDLAT="/mnt/2tbnvS/tmp/h3_v67_def107_1088p_8s.edlat"
OUT_MP4="/mnt/2tbnvS/tmp/h3_v67_def107_1088p_8s.mp4"
OUT_WAV="/mnt/2tbnvS/tmp/h3_v67_def107_1088p_8s.wav"
RESP_JSON="/mnt/2tbnvS/tmp/sample_v67_def107_resp.json"

rm -f "$OUT_EDLAT" "$OUT_MP4" "$OUT_WAV" "$RESP_JSON"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sending /sample request for DEF-107 1088P 8s (6 steps, seed=42)..."
t_start=$(date +%s)
curl -s -X POST http://127.0.0.1:9411/sample \
  -H "Content-Type: application/json" \
  -d "{
    \"latent_path\": \"$IN_EDBIN\",
    \"output_path\": \"$OUT_EDLAT\",
    \"steps\": 6,
    \"seed\": 42
  }" > "$RESP_JSON"

t_end=$(date +%s)
sampling_duration=$((t_end - t_start))
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEF-107 6-Step DiT Sampling finished in $sampling_duration seconds!"
cat "$RESP_JSON"

# 4. Stop server_9411 to free full 8GB VRAM for Video VAE
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stopping server_9411 to free VRAM for Video VAE..."
kill "$SERVER_PID" || pkill -f "server_9411" || true
sleep 2

# 5. Decode .edlat to MP4 and WAV (using DEF-102 concurrent async decoding engine)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Decoding .edlat to MP4 and WAV (Async Concurrent)..."
DECODE_LOG="/mnt/2tbnvS/tmp/decode_v67_def107.log"
/mnt/2tbnvS/h3-rs/target/release/examples/pipeline_e2e_av \
  --input "$OUT_EDLAT" \
  --output "$OUT_MP4" \
  --target both \
  --video-placement ram \
  --audio-device cpu > "$DECODE_LOG" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Video & Audio Decoding Complete: $OUT_MP4"

# 6. Extract 3 evaluation frames (1s, 4s, 7s)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Extracting 3 evaluation frames (1s, 4s, 7s)..."
ffmpeg -y -i "$OUT_MP4" -ss 00:00:01 -vframes 1 /mnt/2tbnvS/tmp/eval_v67_def107_f01.png >/dev/null 2>&1
ffmpeg -y -i "$OUT_MP4" -ss 00:00:04 -vframes 1 /mnt/2tbnvS/tmp/eval_v67_def107_f02.png >/dev/null 2>&1
ffmpeg -y -i "$OUT_MP4" -ss 00:00:07 -vframes 1 /mnt/2tbnvS/tmp/eval_v67_def107_f03.png >/dev/null 2>&1

# 7. Run evaluation script
python3 -c '
import numpy as np
from PIL import Image
import wave

paths = [
    "/mnt/2tbnvS/tmp/eval_v67_def107_f01.png",
    "/mnt/2tbnvS/tmp/eval_v67_def107_f02.png",
    "/mnt/2tbnvS/tmp/eval_v67_def107_f03.png"
]

print("=== 3-POINT PORTRAIT EVALUATION (DEF-107 DUAL-LOOP FUSION 1088P 8S) ===")
for p in paths:
    im = Image.open(p).convert("RGB")
    arr = np.array(im, dtype=np.float32)
    std = float(np.std(arr))
    min_v, max_v = int(np.min(arr)), int(np.max(arr))
    r, g, b = arr[...,0], arr[...,1], arr[...,2]
    skin = (r > 95) & (g > 40) & (b > 20) & ((np.maximum(r, np.maximum(g, b)) - np.minimum(r, np.minimum(g, b))) > 15) & (np.abs(r - g) > 15) & (r > g) & (r > b)
    skin_pct = float(np.mean(skin)) * 100.0
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * gray[1:-1, 1:-1]
    lap_var = float(np.var(lap))
    fname = p.split("/")[-1]
    print(f"[{fname}] std={std:.2f}, dynamic_range=[{min_v}, {max_v}], skin_ratio={skin_pct:.2f}%, laplacian_var={lap_var:.2f}")

wav_path = "/mnt/2tbnvS/tmp/h3_v67_def107_1088p_8s.wav"
with wave.open(wav_path, "rb") as wf:
    n_channels = wf.getnchannels()
    framerate = wf.getframerate()
    n_frames = wf.getnframes()
    raw = wf.readframes(n_frames)
    audio_arr = np.frombuffer(raw, dtype=np.int16)
    peak = int(np.max(np.abs(audio_arr)))
    clip_count = int(np.sum(np.abs(audio_arr) >= 32760))
    clip_pct = (clip_count / len(audio_arr)) * 100.0
    rms = float(np.sqrt(np.mean(audio_arr.astype(np.float64)**2)))
    print(f"=== AUDIO EVALUATION ===")
    print(f"Channels={n_channels}, Rate={framerate}Hz, Duration={n_frames/framerate:.2f}s, Peak={peak}/32767, RMS={rms:.2f}, Clipping={clip_pct:.4f}% ({clip_count} samples)")
'

echo "ALL_V67_DEF107_COMPLETE_SUCCESS"
