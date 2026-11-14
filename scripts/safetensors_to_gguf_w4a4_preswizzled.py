#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
safetensors_to_gguf_w4a4_preswizzled.py
Direct streaming export of Ampere W4A4 pre-swizzled DiT safetensors to GGUF format.
Preserves exact tensor binary bytes (I32 pre-swizzled B fragments, F16 scales, BF16/F32 embeddings/norms).
Zero loss, zero unpacking, O(1) RAM streaming.
"""

import sys
import os
import json
import struct
import time

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_I8 = 16
GGML_TYPE_I16 = 17
GGML_TYPE_I32 = 18
GGML_TYPE_I64 = 19
GGML_TYPE_F64 = 20
GGML_TYPE_BF16 = 29

GGUF_METADATA_TYPE_UINT8 = 0
GGUF_METADATA_TYPE_INT8 = 1
GGUF_METADATA_TYPE_UINT16 = 2
GGUF_METADATA_TYPE_INT16 = 3
GGUF_METADATA_TYPE_UINT32 = 4
GGUF_METADATA_TYPE_INT32 = 5
GGUF_METADATA_TYPE_FLOAT32 = 6
GGUF_METADATA_TYPE_BOOL = 7
GGUF_METADATA_TYPE_STRING = 8
GGUF_METADATA_TYPE_ARRAY = 9
GGUF_METADATA_TYPE_UINT64 = 10
GGUF_METADATA_TYPE_INT64 = 11
GGUF_METADATA_TYPE_FLOAT64 = 12

def map_dtype(st_dtype):
    mapping = {
        "F32": GGML_TYPE_F32,
        "F16": GGML_TYPE_F16,
        "BF16": GGML_TYPE_BF16,
        "I32": GGML_TYPE_I32,
        "I8": GGML_TYPE_I8,
        "U8": GGML_TYPE_I8,
    }
    if st_dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {st_dtype}")
    return mapping[st_dtype]

def write_string(f, s):
    b = s.encode('utf-8')
    f.write(struct.pack('<Q', len(b)))
    f.write(b)

def write_kv_string(f, k, v):
    write_string(f, k)
    f.write(struct.pack('<I', GGUF_METADATA_TYPE_STRING))
    write_string(f, v)

def write_kv_uint32(f, k, v):
    write_string(f, k)
    f.write(struct.pack('<I', GGUF_METADATA_TYPE_UINT32))
    f.write(struct.pack('<I', v))

def convert(src_path, dst_path):
    t0 = time.time()
    print(f"[1/4] Reading safetensors header from {src_path}...")
    with open(src_path, 'rb') as f_src:
        header_len = int.from_bytes(f_src.read(8), 'little')
        header_json = json.loads(f_src.read(header_len).decode('utf-8'))
        data_base_offset = 8 + header_len
        
        meta = header_json.get('__metadata__', {})
        tensors = {k: v for k, v in header_json.items() if k != '__metadata__'}
        print(f"      Total tensors: {len(tensors)}, Safetensors header size: {header_len} bytes")
        
        # Metadata KV pairs
        kv_pairs = [
            ("general.architecture", "minimax_h3", GGUF_METADATA_TYPE_STRING),
            ("general.alignment", 32, GGUF_METADATA_TYPE_UINT32),
            ("general.file_type", 1, GGUF_METADATA_TYPE_UINT32),
            ("h3.quant.format", "ampere_w4a4_group256_preswizzled", GGUF_METADATA_TYPE_STRING),
            ("h3.quant.group_size", 256, GGUF_METADATA_TYPE_UINT32),
            ("weights", "sym_w4a4_convrot", GGUF_METADATA_TYPE_STRING),
            ("w4a4", "1", GGUF_METADATA_TYPE_STRING),
        ]
        
        # Include original metadata keys as strings if not already set
        for mk, mv in meta.items():
            if mk not in [k for k, _, _ in kv_pairs]:
                kv_pairs.append((mk, str(mv), GGUF_METADATA_TYPE_STRING))
        
        alignment = 32
        tensor_names = sorted(tensors.keys())
        tensor_infos = []
        cur_offset = 0
        
        print(f"[2/4] Planning contiguous 32-byte aligned tensor layout...")
        for name in tensor_names:
            t = tensors[name]
            shape = t['shape']
            st_dtype = t['dtype']
            ggml_type = map_dtype(st_dtype)
            start, end = t['data_offsets']
            size = end - start
            
            rev_dims = list(reversed(shape)) if len(shape) > 0 else [1]
            tensor_infos.append({
                'name': name,
                'dims': rev_dims,
                'ggml_type': ggml_type,
                'offset': cur_offset,
                'src_start': start,
                'size': size
            })
            pad = (alignment - (size % alignment)) % alignment
            cur_offset += size + pad
            
        print(f"[3/4] Writing GGUF header to {dst_path}...")
        tmp_dst = dst_path + ".tmp"
        with open(tmp_dst, 'wb') as f_dst:
            f_dst.write(b'GGUF')
            f_dst.write(struct.pack('<I', 3)) # GGUF v3
            f_dst.write(struct.pack('<Q', len(tensor_infos)))
            f_dst.write(struct.pack('<Q', len(kv_pairs)))
            
            for k, val, vtype in kv_pairs:
                if vtype == GGUF_METADATA_TYPE_STRING:
                    write_kv_string(f_dst, k, val)
                elif vtype == GGUF_METADATA_TYPE_UINT32:
                    write_kv_uint32(f_dst, k, val)
                    
            for ti in tensor_infos:
                write_string(f_dst, ti['name'])
                n_dims = len(ti['dims'])
                f_dst.write(struct.pack('<I', n_dims))
                for d in ti['dims']:
                    f_dst.write(struct.pack('<Q', d))
                f_dst.write(struct.pack('<I', ti['ggml_type']))
                f_dst.write(struct.pack('<Q', ti['offset']))
                
            header_end = f_dst.tell()
            pad_header = (alignment - (header_end % alignment)) % alignment
            if pad_header > 0:
                f_dst.write(b'\x00' * pad_header)
                
            data_start = f_dst.tell()
            print(f"      Header written. Data section starts at byte {data_start}.")
            print(f"[4/4] Streaming {len(tensor_infos)} tensor payloads (chunk: 4MB)...")
            
            for i, ti in enumerate(tensor_infos):
                f_src.seek(data_base_offset + ti['src_start'])
                remaining = ti['size']
                while remaining > 0:
                    chunk_sz = min(remaining, 4 * 1024 * 1024)
                    buf = f_src.read(chunk_sz)
                    f_dst.write(buf)
                    remaining -= chunk_sz
                pad = (alignment - (ti['size'] % alignment)) % alignment
                if pad > 0:
                    f_dst.write(b'\x00' * pad)
                    
                if (i + 1) % 150 == 0 or (i + 1) == len(tensor_infos):
                    elapsed = time.time() - t0
                    print(f"      Streamed {i + 1}/{len(tensor_infos)} tensors ({elapsed:.1f}s)...")
                    
        os.rename(tmp_dst, dst_path)
        total_size = os.path.getsize(dst_path)
        t_total = time.time() - t0
        print(f"SUCCESS: Exported {total_size / (1024**3):.2f} GB GGUF in {t_total:.2f}s ({total_size / (1024**2) / t_total:.1f} MB/s)")

if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.safetensors"
    dst = sys.argv[2] if len(sys.argv) > 2 else "/mnt/2tbnvS/models/minimax_h3_fl2va_hyperflow_flash_14b_w4a4_preswizzled.gguf"
    convert(src, dst)
