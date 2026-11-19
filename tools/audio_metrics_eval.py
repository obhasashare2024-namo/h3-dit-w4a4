#!/usr/bin/env python3
import sys
import numpy as np
import soundfile as sf
from scipy import signal

def analyze_audio(wav_path):
    data, sr = sf.read(wav_path)
    if data.ndim > 1:
        mono = np.mean(data, axis=1)
    else:
        mono = data

    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono**2)))
    dc_offset = float(np.mean(mono))
    
    clip_mask = np.abs(mono) >= 0.999
    point_clip_pct = float(np.mean(clip_mask) * 100.0)
    
    runs = 0
    consecutive = 0
    for c in clip_mask:
        if c:
            consecutive += 1
            if consecutive == 3:
                runs += 1
        else:
            consecutive = 0

    freqs, psd = signal.welch(mono, sr, nperseg=2048)
    total_energy = float(np.sum(psd))
    energy_gt_8k = float(np.sum(psd[freqs >= 8000]))
    energy_gt_12k = float(np.sum(psd[freqs >= 12000]))
    ratio_8k = (energy_gt_8k / (total_energy + 1e-12)) * 100.0
    ratio_12k = (energy_gt_12k / (total_energy + 1e-12)) * 100.0

    geom_mean = np.exp(np.mean(np.log(psd + 1e-12)))
    arith_mean = np.mean(psd)
    spectral_flatness = float(geom_mean / (arith_mean + 1e-12))

    return {
        "sr": sr,
        "samples": len(mono),
        "peak": peak,
        "rms": rms,
        "dc_offset": dc_offset,
        "point_clip_pct": point_clip_pct,
        "continuous_clip_runs": runs,
        "energy_gt_8k_pct": ratio_8k,
        "energy_gt_12k_pct": ratio_12k,
        "spectral_flatness": spectral_flatness,
    }

def main():
    targets = [
        ("Golden Reference", "/mnt/2tbnvS/tmp/h3_render_1080p_3s_uncompressed.wav"),
        ("W4A8 Mixed", "/mnt/2tbnvS/tmp/h3_w4a8_e2e_3s.wav"),
        ("W4A4 Preswizzled", "/mnt/2tbnvS/tmp/h3_w4a4_e2e_3s.wav"),
        ("GGUF Q3_K_M", "/mnt/2tbnvS/tmp/h3_q3km_e2e_3s.wav"),
    ]

    print("=" * 95)
    print(f"{'Model':<20} | {'Peak':<6} | {'RMS':<6} | {'Clip%':<7} | {'Runs>=3':<7} | {'>8kHz%':<7} | {'>12kHz%':<7} | {'Flatness':<8}")
    print("=" * 95)

    for name, path in targets:
        try:
            st = analyze_audio(path)
            print(f"{name:<20} | {st['peak']:.4f} | {st['rms']:.4f} | {st['point_clip_pct']:6.3f}% | {st['continuous_clip_runs']:<7} | {st['energy_gt_8k_pct']:6.2f}% | {st['energy_gt_12k_pct']:6.2f}% | {st['spectral_flatness']:.6f}")
        except Exception as e:
            print(f"{name:<20} | ERROR: {e}")

    print("=" * 95)

if __name__ == '__main__':
    main()
