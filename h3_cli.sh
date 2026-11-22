#!/usr/bin/env bash
set -e

# ==============================================================================
# MiniMax H3 Unified Master CLI: Multi-Resolution & Flexible Step Engine
# Supports:
#   - 768p-to-1080p (Ultra Fast ~2min with Lanczos+Unsharp Detail Enhancement)
#   - 768p-to-2k    (Ultra Fast ~2min 2K 2560x1440 Lanczos)
#   - 768p-native   (Native 768x448 ~2min)
#   - 1088p-native  (Native 1920x1088 Full Dense FA-2, Perfect Long-Range Lock)
#   - 1088p-hybrid  (Native 1920x1088 Step 1 Dense + Steps 2..N Win8192)
# Step Options: 4, 6, 8, or custom
# ==============================================================================

RES="768p-to-1080p"
STEPS=6
SEED=42
OUTPUT="/mnt/2tbnvS/tmp/h3_render_cli_out.mp4"
PORT=9411
EVAL=true

while [[ $# -gt 0 ]]; do
  case $1 in
    --res|--resolution)
      RES="$2"; shift 2 ;;
    --steps)
      STEPS="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --output|-o)
      OUTPUT="$2"; shift 2 ;;
    --port)
      PORT="$2"; shift 2 ;;
    --no-eval)
      EVAL=false; shift 1 ;;
    *)
      echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

echo "================================================================================"
echo "  MiniMax H3 Multi-Modal Render CLI"
echo "  Resolution Mode : $RES"
echo "  Sampling Steps  : $STEPS"
echo "  TorchGen Seed   : $SEED"
echo "  Output Media    : $OUTPUT"
echo "================================================================================"

case $RES in
  768p-to-1080p)
    LATENT_IN="/mnt/2tbnvS/tmp/stage1_input_768p_5s.edbin"
    SCALE_OPT="1080p"
    HYBRID_ENV=""
    ;;
  768p-to-2k)
    LATENT_IN="/mnt/2tbnvS/tmp/stage1_input_768p_5s.edbin"
    SCALE_OPT="2k"
    HYBRID_ENV=""
    ;;
  768p-native)
    LATENT_IN="/mnt/2tbnvS/tmp/stage1_input_768p_5s.edbin"
    SCALE_OPT="none"
    HYBRID_ENV=""
    ;;
  1088p-native|1088p-dense)
    LATENT_IN="/mnt/2tbnvS/tmp/stage1_input_1080p_8s.edbin"
    SCALE_OPT="none"
    HYBRID_ENV=""
    ;;
  1088p-hybrid)
    LATENT_IN="/mnt/2tbnvS/tmp/stage1_input_1080p_8s.edbin"
    SCALE_OPT="none"
    HYBRID_ENV="8192"
    ;;
  *)
    echo "Error: Invalid --res option '$RES'." >&2
    echo "Available: 768p-to-1080p, 768p-to-2k, 768p-native, 1088p-native, 1088p-hybrid" >&2
    exit 1
    ;;
esac

# 1. Start server_9411 if not already running on target port
SERVER_PID=""
if ! curl -s http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
    echo "[1/4] Starting server_$PORT..."
    pkill -f "server_$PORT" || true
    sleep 1

    export H3_ATTN="flash"
    export H3_ATTN_GROUPS="14"
    export H3_STREAMING="1"
    export H3_FORCE_GPU_ATT="0"
    export H3_ZERO_OFFLOAD="1"
    if [[ -n "$HYBRID_ENV" ]]; then
        export H3_HYBRID_WINDOW="$HYBRID_ENV"
    else
        unset H3_HYBRID_WINDOW
    fi
    unset H3_STEP0_PREFIX

    /mnt/2tbnvS/h3-rs/target/release/server_9411       --port $PORT       --model /mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.safetensors       --vram-blocks 0       --ram-blocks 50       --token-chunk 4096       --mlp-chunk 2048       --attn flash       > "/mnt/2tbnvS/tmp/server_${PORT}_cli.log" 2>&1 &
    SERVER_PID=$!

    for i in {1..60}; do
        if curl -s http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
            echo "      server_$PORT is READY!"
            break
        fi
        sleep 1
    done
else
    echo "[1/4] server_$PORT already active."
fi

# 2. Sample latents
TMP_EDLAT="${OUTPUT%.mp4}.edlat"
rm -f "$TMP_EDLAT"
RESP_JSON="/mnt/2tbnvS/tmp/h3_cli_resp.json"

echo "[2/4] Sampling DiT ($STEPS steps, seed=$SEED)..."
t_sample_start=$(date +%s%N)
curl -s -X POST http://127.0.0.1:$PORT/sample   -H "Content-Type: application/json"   -d "{
    \"latent_path\": \"$LATENT_IN\",
    \"output_path\": \"$TMP_EDLAT\",
    \"steps\": $STEPS,
    \"seed\": $SEED
  }" > "$RESP_JSON"
t_sample_end=$(date +%s%N)
sample_ms=$(( (t_sample_end - t_sample_start) / 1000000 ))
sample_sec=$(echo "scale=2; $sample_ms / 1000" | bc)
echo "      DiT Sampling completed in ${sample_sec}s!"
cat "$RESP_JSON"
echo ""

# 3. Stop server to free VRAM for Video VAE
if [[ -n "$SERVER_PID" ]]; then
    echo "[3/4] Stopping temporary server to free 8GB VRAM for Video VAE..."
    kill "$SERVER_PID" || pkill -f "server_$PORT" || true
    sleep 2
else
    pkill -f "server_$PORT" || true
    sleep 2
fi

# 4. Decode Video & Audio (with scaling / unsharp if requested)
echo "[4/4] Decoding & Rendering to $OUTPUT (scale: $SCALE_OPT)..."
t_dec_start=$(date +%s%N)
/mnt/2tbnvS/h3-rs/target/release/examples/pipeline_e2e_av   --input "$TMP_EDLAT"   --output "$OUTPUT"   --target both   --scale "$SCALE_OPT"   --video-placement ram   --audio-device cpu
t_dec_end=$(date +%s%N)
dec_ms=$(( (t_dec_end - t_dec_start) / 1000000 ))
dec_sec=$(echo "scale=2; $dec_ms / 1000" | bc)
total_sec=$(echo "scale=2; $sample_sec + $dec_sec" | bc)

echo "================================================================================"
echo "🎉 Generation Finished!"
echo "   Resolution Mode: $RES"
echo "   Output File    : $OUTPUT"
echo "   DiT Sampling   : ${sample_sec}s"
echo "   Decode & Render: ${dec_sec}s"
echo "   Total E2E Time : ${total_sec}s"
echo "================================================================================"

if [ "$EVAL" = true ]; then
    echo "Running 3-Point Portrait & Audio Quality Evaluation..."
    F01="${OUTPUT%.mp4}_f01.png"
    F02="${OUTPUT%.mp4}_f02.png"
    F03="${OUTPUT%.mp4}_f03.png"
    ffmpeg -y -i "$OUTPUT" -ss 00:00:01 -vframes 1 "$F01" >/dev/null 2>&1 || true
    ffmpeg -y -i "$OUTPUT" -ss 00:00:03 -vframes 1 "$F02" >/dev/null 2>&1 || true
    ffmpeg -y -i "$OUTPUT" -ss 00:00:04.5 -vframes 1 "$F03" >/dev/null 2>&1 || true

    python3 -c '
import numpy as np
from PIL import Image
import wave, os, sys

paths = ["'"$F01"'", "'"$F02"'", "'"$F03"'"]
print("=== PORTRAIT QUALITY EVALUATION ===")
for p in paths:
    if not os.path.exists(p): 
        print(f"[MISSING] {p}")
        continue
    im = Image.open(p).convert("RGB")
    arr = np.array(im, dtype=np.float32)
    std = float(np.std(arr))
    r, g, b = arr[...,0], arr[...,1], arr[...,2]
    skin = (r > 95) & (g > 40) & (b > 20) & ((np.maximum(r, np.maximum(g, b)) - np.minimum(r, np.minimum(g, b))) > 15) & (np.abs(r - g) > 15) & (r > g) & (r > b)
    skin_pct = float(np.mean(skin)) * 100.0
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * gray[1:-1, 1:-1]
    lap_var = float(np.var(lap))
    print(f"[{os.path.basename(p)}] Res={im.size}, Std={std:.2f}, Skin={skin_pct:.2f}%, LapVar={lap_var:.2f}")

wav_p = "'"$OUTPUT"'".replace(".mp4", ".wav")
if os.path.exists(wav_p):
    with wave.open(wav_p, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        audio_arr = np.frombuffer(raw, dtype=np.int16)
        clip_count = int(np.sum(np.abs(audio_arr) >= 32760))
        clip_pct = (clip_count / len(audio_arr)) * 100.0
        print(f"=== AUDIO EVALUATION ===")
        print(f"Channels={wf.getnchannels()}, Rate={wf.getframerate()}Hz, Peak={np.max(np.abs(audio_arr))}/32767, Clipping={clip_pct:.4f}%")
'
fi
