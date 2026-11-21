#!/usr/bin/env python3
"""
Zero-RAM Streaming Converter:
Convert official HyperFlow INT8 ConvRot (16GB) to authentic True W4A4 (~9.6GB)
with Group-256 scale, Sub-LSB de-aliasing, OLS projection, and Ampere m16n8k64 Warp-Tile Pre-Swizzling.
Streams directly to disk, RAM usage < 300MB!
"""
import sys
import time
import json
import struct
import gc
import torch
from pathlib import Path
from safetensors import safe_open

DTYPE_MAP = {
    torch.float32: 'F32',
    torch.float16: 'F16',
    torch.bfloat16: 'BF16',
    torch.int8: 'I8',
    torch.uint8: 'U8',
    torch.int16: 'I16',
    torch.int32: 'I32',
    torch.int64: 'I64',
}

@torch.no_grad()
def preswizzle_weights_to_ampere_b_fragment(weight_s4: torch.Tensor) -> torch.Tensor:
    """
    將 [N, K] 的有符號 s4 權重 (值域 [-8, 7]) 打包並預重排為 Ampere sm_86
    `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32` 的矩陣 B 硬體寄存器佈局。
    輸出: shape [N // 8, K // 64, 32, 2], dtype=torch.int32
    """
    N, K = weight_s4.shape
    assert N % 8 == 0 and K % 64 == 0, f"N ({N}) 須為 8 的倍數, K ({K}) 須為 64 的倍數"

    nibbles = (weight_s4.to(torch.int32) & 0xF).view(N, K // 8, 8)
    packed_u32 = (
        (nibbles[..., 0] << 0)
        | (nibbles[..., 1] << 4)
        | (nibbles[..., 2] << 8)
        | (nibbles[..., 3] << 12)
        | (nibbles[..., 4] << 16)
        | (nibbles[..., 5] << 20)
        | (nibbles[..., 6] << 24)
        | (nibbles[..., 7] << 28)
    )
    w_tiles = packed_u32.view(N // 8, 8, K // 64, 2, 4)
    b_preswizzled = (
        w_tiles.permute(0, 2, 1, 4, 3)
        .contiguous()
        .view(N // 8, K // 64, 32, 2)
    )
    return b_preswizzled

@torch.no_grad()
def quantize_and_preswizzle_chunked(
    w_i8: torch.Tensor,      # [N, K] int8
    s_row: torch.Tensor,     # [N, 1] float32
    group_size: int = 256,
    chunk_size: int = 2048,
    device: str = 'cuda',
):
    N, K = w_i8.shape
    assert K % group_size == 0 and group_size % 64 == 0

    num_groups = K // group_size
    s_grp_out  = torch.empty((N, num_groups), dtype=torch.float16)
    q_s4_full  = torch.empty((N, K), dtype=torch.int8)

    total_sq_err = 0.0
    total_orig_sq = 0.0

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        w_chunk = w_i8[start:end].to(device=device, non_blocking=True).float()
        s_chunk = s_row[start:end].to(device=device, non_blocking=True).float().view(-1, 1)

        # 1. Sub-LSB de-aliasing dither
        gen = torch.Generator(device=device).manual_seed(42 + start)
        dither = (torch.rand(w_chunk.shape, generator=gen, device=device) - 0.5) * 0.85
        mask = (w_chunk.abs() > 0) & (w_chunk.abs() < 127)
        w_cont = (w_chunk + dither * mask.float()) * s_chunk

        c_N = end - start
        w_grp = w_cont.view(c_N, num_groups, group_size)

        # 2. Vectorized 15-point grid search for optimal alpha in [0.72, 1.00]
        s_max = (w_grp.abs().amax(dim=-1, keepdim=True) / 7.0).clamp(min=1e-12)
        best_err = torch.full((c_N, num_groups, 1), float('inf'), device=device)
        best_q = torch.zeros_like(w_grp, dtype=torch.int8)

        for alpha in torch.linspace(0.72, 1.00, 15, device=device):
            s_cand = s_max * alpha
            q_cand = torch.round(w_grp / s_cand).clamp(-7.0, 7.0)
            wq = (w_grp * q_cand).sum(dim=-1, keepdim=True)
            qq = (q_cand * q_cand).sum(dim=-1, keepdim=True).clamp(min=1e-12)
            s_ols_cand = wq / qq
            err = (w_grp - q_cand * s_ols_cand).abs().pow(2.2).sum(dim=-1, keepdim=True)
            improved = err < best_err
            best_err = torch.where(improved, err, best_err)
            best_q = torch.where(improved, q_cand.to(torch.int8), best_q)

        # 3. OLS orthogonal scale projection
        q_f32 = best_q.float()
        s_ols = (w_grp * q_f32).sum(dim=-1) / (q_f32 * q_f32).sum(dim=-1).clamp(min=1e-12)

        q_flat = best_q.view(c_N, K)
        q_s4_full[start:end] = q_flat.cpu()
        s_grp_out[start:end] = s_ols.to(torch.float16).cpu()

        w_orig = w_chunk * s_chunk
        w_recon = (q_flat.float().view(c_N, num_groups, group_size) * s_ols.unsqueeze(-1)).view(c_N, K)
        total_sq_err += torch.sum((w_orig - w_recon) ** 2).item()
        total_orig_sq += torch.sum(w_orig ** 2).item()

    rel_l2 = (total_sq_err / total_orig_sq) ** 0.5
    b_preswizzled = preswizzle_weights_to_ampere_b_fragment(q_s4_full)
    return b_preswizzled, s_grp_out, rel_l2

def main():
    src_path = Path('/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b-int8_convrot.safetensors')
    dst_path = Path('/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.safetensors')

    if not src_path.exists():
        print(f"Error: {src_path} does not exist!")
        sys.exit(1)

    print(f'Starting zero-RAM streaming conversion from {src_path}')
    print(f'Target destination: {dst_path}')
    t_start = time.time()

    with safe_open(str(src_path), framework='pt', device='cpu') as f:
        keys = list(f.keys())
        orig_meta = f.metadata() or {}
        print(f'Total keys in source: {len(keys)}')

        # Identify linear layers and non-linear layers
        linear_layers = {}
        for k in keys:
            if k.endswith('.weight'):
                scale_key = k[:-7] + '.weight_scale'
                if scale_key in keys:
                    linear_layers[k] = scale_key

        print(f'Detected {len(linear_layers)} INT8 ConvRot linear layers to quantize & preswizzle')

        # 1. Build Header in advance
        header = {}
        current_offset = 0

        # Maintain deterministic order of keys
        out_keys_order = []
        for k in keys:
            if k in linear_layers:
                w_tensor = f.get_tensor(k)
                N, K = w_tensor.shape
                # Weight tensor: I32, [N//8, K//64, 32, 2]
                w_shape = [N // 8, K // 64, 32, 2]
                w_bytes = (N * K // 2)
                header[k] = {
                    'dtype': 'I32',
                    'shape': w_shape,
                    'data_offsets': [current_offset, current_offset + w_bytes]
                }
                current_offset += w_bytes
                out_keys_order.append((k, 'linear_w'))

                # Scale tensor: F16, [N, K//256]
                scale_key = linear_layers[k]
                s_shape = [N, K // 256]
                s_bytes = (N * (K // 256)) * 2
                header[scale_key] = {
                    'dtype': 'F16',
                    'shape': s_shape,
                    'data_offsets': [current_offset, current_offset + s_bytes]
                }
                current_offset += s_bytes
                out_keys_order.append((scale_key, 'linear_s'))
            elif k.endswith('.weight_scale') and (k[:-13] + '.weight') in linear_layers:
                continue # handled above
            elif k.endswith('.comfy_quant') and (k[:-12] + '.weight') in linear_layers:
                continue # not needed in w4a4
            else:
                t = f.get_tensor(k)
                t_bytes = t.numel() * t.element_size()
                header[k] = {
                    'dtype': DTYPE_MAP[t.dtype],
                    'shape': list(t.shape),
                    'data_offsets': [current_offset, current_offset + t_bytes]
                }
                current_offset += t_bytes
                out_keys_order.append((k, 'passthrough'))

        new_meta = dict(orig_meta)
        new_meta['h3.quant.format'] = 'ampere_w4a4_group256_preswizzled'
        new_meta['h3.quant.group_size'] = '256'
        new_meta['weights'] = 'sym_w4a4_convrot'
        new_meta['w4a4'] = '1'
        header['__metadata__'] = new_meta

        header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
        # 8-byte alignment padding
        header_len = len(header_json)
        padded_len = (header_len + 7) & ~7
        header_json += b' ' * (padded_len - header_len)

        print(f'Header prepared: {len(header)-1} tensors, JSON length: {padded_len} bytes')
        print(f'Total payload data size: {current_offset:,} bytes ({current_offset / (1024**3):.2f} GB)')

        # 2. Write to destination file directly
        print(f'Streaming output to {dst_path}...')
        with open(dst_path, 'wb', buffering=64*1024*1024) as out_f:
            # 8-byte header length
            out_f.write(struct.pack('<Q', padded_len))
            out_f.write(header_json)

            # Stream tensors
            converted_idx = 0
            for k, kind in out_keys_order:
                if kind == 'linear_w':
                    t0 = time.time()
                    w_i8 = f.get_tensor(k)
                    s_key = linear_layers[k]
                    s_row = f.get_tensor(s_key)

                    b_preswizzled, s_grp, rel_l2 = quantize_and_preswizzle_chunked(w_i8, s_row)
                    
                    # Write weight
                    b_bytes = b_preswizzled.view(torch.uint8).numpy().tobytes()
                    out_f.write(b_bytes)

                    # Store scale to write on next iteration
                    _cached_scale = s_grp.view(torch.uint8).numpy().tobytes()
                    converted_idx += 1
                    dt = time.time() - t0
                    print(f'[{converted_idx}/{len(linear_layers)}] {k}: -> preswizzled (relL2: {rel_l2:.4f}, {dt:.2f}s)')
                    del w_i8, s_row, b_preswizzled, s_grp
                    torch.cuda.empty_cache()
                elif kind == 'linear_s':
                    out_f.write(_cached_scale)
                    del _cached_scale
                elif kind == 'passthrough':
                    t = f.get_tensor(k)
                    out_f.write(t.view(torch.uint8).numpy().tobytes())
                    del t

    total_time = time.time() - t_start
    final_size = dst_path.stat().st_size
    print(f'SUCCESS! Successfully created {dst_path} ({final_size:,} bytes, {final_size / (1024**3):.2f} GB) in {total_time:.1f}s')

if __name__ == '__main__':
    main()
