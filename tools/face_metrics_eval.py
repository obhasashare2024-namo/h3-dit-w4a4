#!/usr/bin/env python3
import sys
import os
import subprocess
import json
import numpy as np

def extract_frames(video_path, num_frames=8):
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_frames,duration',
        '-of', 'json', video_path
    ]
    p = subprocess.run(probe_cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path}: {p.stderr}")
    meta = json.loads(p.stdout)['streams'][0]
    w = int(meta['width'])
    h = int(meta['height'])

    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vframes', str(num_frames),
        '-f', 'image2pipe',
        '-pix_fmt', 'rgb24',
        '-vcodec', 'rawvideo', '-'
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw = proc.stdout.read()
    proc.stdout.close()
    proc.wait()

    frame_bytes = w * h * 3
    n = len(raw) // frame_bytes
    if n == 0:
        raise RuntimeError(f"Failed to extract frames from {video_path}")

    frames = np.frombuffer(raw[:n * frame_bytes], dtype=np.uint8).reshape((n, h, w, 3))
    return frames, w, h, meta

def compute_metrics(frames):
    n, h, w, _ = frames.shape
    rgb_stds = []
    lap_vars = []
    center_lap_vars = []

    # Center crop for portrait / facial ROI (central 40% height, central 40% width)
    ch_start, ch_end = int(h * 0.3), int(h * 0.7)
    cw_start, cw_end = int(w * 0.3), int(w * 0.7)

    for i in range(n):
        f = frames[i]
        rgb_stds.append(float(np.std(f)))

        gray = np.dot(f[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
        # 2D Laplacian operator
        lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * gray[1:-1, 1:-1]
        lap_vars.append(float(np.var(lap)))

        # Center crop Laplacian (facial region)
        gray_c = gray[ch_start:ch_end, cw_start:cw_end]
        lap_c = gray_c[:-2, 1:-1] + gray_c[2:, 1:-1] + gray_c[1:-1, :-2] + gray_c[1:-1, 2:] - 4.0 * gray_c[1:-1, 1:-1]
        center_lap_vars.append(float(np.var(lap_c)))

    # Temporal flicker (mean absolute difference between adjacent frames)
    temporal_diffs = []
    for i in range(1, n):
        diff = np.mean(np.abs(frames[i].astype(np.float32) - frames[i-1].astype(np.float32)))
        temporal_diffs.append(float(diff))
    mean_temp_diff = float(np.mean(temporal_diffs)) if temporal_diffs else 0.0

    return {
        "num_frames": n,
        "resolution": f"{w}x{h}",
        "rgb_std_mean": float(np.mean(rgb_stds)),
        "rgb_std_min": float(np.min(rgb_stds)),
        "lap_var_mean": float(np.mean(lap_vars)),
        "center_lap_var_mean": float(np.mean(center_lap_vars)),
        "temporal_diff_mean": mean_temp_diff,
    }

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <test_video.mp4> [golden_ref_video.mp4]")
        sys.exit(1)

    test_path = sys.argv[1]
    golden_path = sys.argv[2] if len(sys.argv) > 2 else "/mnt/2tbnvS/tmp/h3_q3km_render_1080p_8s.mp4"

    print("=" * 80)
    print("=== MiniMax-H3 Face & Video Preservation Gate Evaluator (rc001 Gate) ===")
    print("=" * 80)
    print(f"Test Video   : {test_path}")
    print(f"Golden Ref   : {golden_path}")

    frames_test, wt, ht, _ = extract_frames(test_path)
    metrics_test = compute_metrics(frames_test)

    has_ref = os.path.exists(golden_path)
    if has_ref:
        frames_ref, wr, hr, _ = extract_frames(golden_path)
        metrics_ref = compute_metrics(frames_ref)
    else:
        metrics_ref = None

    print("-" * 80)
    print(f"{'Metric':<30} | {'Test Video':<18} | {'Golden Ref':<18} | {'Gate / Verdict':<15}")
    print("-" * 80)

    # 1. RGB std
    std_pass = metrics_test['rgb_std_mean'] >= 50.0
    ref_std_str = f"{metrics_ref['rgb_std_mean']:.2f}" if metrics_ref else "N/A"
    print(f"{'RGB Std (Mean)':<30} | {metrics_test['rgb_std_mean']:<18.2f} | {ref_std_str:<18} | {'[PASS >= 50]' if std_pass else '[FAIL < 50]'}")

    # 2. Laplacian Variance (Full frame)
    ref_lap_str = f"{metrics_ref['lap_var_mean']:.2f}" if metrics_ref else "N/A"
    print(f"{'Laplacian Var (Full)':<30} | {metrics_test['lap_var_mean']:<18.2f} | {ref_lap_str:<18} | {'INFO'}")

    # 3. Facial Center Laplacian Ratio
    if metrics_ref:
        c_ratio = metrics_test['center_lap_var_mean'] / (metrics_ref['center_lap_var_mean'] + 1e-12)
        c_pass = 0.90 <= c_ratio <= 1.15
        ref_c_str = f"{metrics_ref['center_lap_var_mean']:.2f}"
        print(f"{'Facial Laplacian Var':<30} | {metrics_test['center_lap_var_mean']:<18.2f} | {ref_c_str:<18} | {f'Ratio: {c_ratio:.3f}'}")
        print(f"{'Facial Laplacian Ratio':<30} | {c_ratio:<18.3f} | {'1.000 (Target)':<18} | {'[PASS]' if c_pass else '[FAIL out [0.9,1.15]]'}")
    else:
        print(f"{'Facial Laplacian Var':<30} | {metrics_test['center_lap_var_mean']:<18.2f} | {'N/A':<18} | {'N/A'}")

    # 4. Temporal Jitter / Flicker
    if metrics_ref:
        temp_ratio = metrics_test['temporal_diff_mean'] / (metrics_ref['temporal_diff_mean'] + 1e-12)
        temp_pass = temp_ratio <= 1.20
        ref_t_str = f"{metrics_ref['temporal_diff_mean']:.2f}"
        print(f"{'Temporal Flicker (Diff)':<30} | {metrics_test['temporal_diff_mean']:<18.2f} | {ref_t_str:<18} | {f'Ratio: {temp_ratio:.3f}'}")
        print(f"{'Temporal Stability Ratio':<30} | {temp_ratio:<18.3f} | {'<= 1.200 (Max)':<18} | {'[PASS]' if temp_pass else '[FAIL > 1.20]'}")
    else:
        print(f"{'Temporal Flicker (Diff)':<30} | {metrics_test['temporal_diff_mean']:<18.2f} | {'N/A':<18} | {'N/A'}")

    print("=" * 80)

if __name__ == '__main__':
    main()
