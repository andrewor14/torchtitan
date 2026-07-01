"""
Convert GPT-OSS-20B checkpoint from MXFP4 to NVFP4 format.

MXFP4 (OCP MX): E2M1 FP4 values with E8M0 block scales, block_size=32
NVFP4 (NVIDIA): E2M1 FP4 values with FP8-E4M3 block scales + FP32 global scale, block_size=16

Only MoE expert weights are quantized; attention, embeddings, router stay bf16.

Usage:
    python scripts/convert_gptoss_mxfp4_to_nvfp4.py \
        --input-dir ~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/ \
        --output-dir /data/users/andrewor/checkpoints/gpt-oss-20b-nvfp4/
"""

import argparse
import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# FP4 E2M1 lookup table (4 bits: 1 sign + 2 exponent + 1 mantissa)
# Index = 4-bit value, value = float representation
FP4_E2M1_LUT = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,    # positive (sign=0)
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # negative (sign=1)
], dtype=torch.float32)

FP4_MAX = 6.0
FP8_MAX = 448.0
NVFP4_BLOCK_SIZE = 16


def dequantize_mxfp4(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """
    Dequantize MXFP4 packed data to bf16.

    Args:
        blocks: [E, out_features, num_blocks, 16] uint8 — 16 bytes = 32 FP4 values per block
        scales: [E, out_features, num_blocks] uint8 — E8M0 scale per block

    Returns:
        bf16 tensor [E, out_features, in_features] where in_features = num_blocks * 32
    """
    E, out_features, num_blocks, bytes_per_block = blocks.shape
    assert bytes_per_block == 16, f"Expected 16 bytes per block, got {bytes_per_block}"

    # Unpack: 2 FP4 values per byte (low nibble first, then high nibble)
    low_nibble = (blocks & 0x0F).to(torch.int64)
    high_nibble = ((blocks >> 4) & 0x0F).to(torch.int64)
    # Interleave: [low0, high0, low1, high1, ...]
    unpacked = torch.stack([low_nibble, high_nibble], dim=-1)
    unpacked = unpacked.reshape(E, out_features, num_blocks, 32)

    # Lookup FP4 values
    fp4_values = FP4_E2M1_LUT[unpacked]  # [E, out_features, num_blocks, 32]

    # E8M0 scale: value = 2^(exponent - 127)
    e8m0_scale = torch.pow(
        2.0,
        scales.to(torch.float32) - 127.0
    ).unsqueeze(-1)  # [E, out_features, num_blocks, 1]

    # Dequantize
    dequantized = fp4_values * e8m0_scale  # [E, out_features, num_blocks, 32]

    # Flatten blocks dimension
    in_features = num_blocks * 32
    return dequantized.reshape(E, out_features, in_features).to(torch.bfloat16)


def quantize_nvfp4(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize bf16 weight to NVFP4 format (per-expert).

    Args:
        weight: [E, out_features, in_features] bf16

    Returns:
        packed: [E, out_features, in_features // 2] uint8 — 2 FP4 values per byte
        block_scales: [E, out_features, in_features // 16] float8_e4m3fn
        global_scales: [E] float32 — per-expert global scale
    """
    E, out_features, in_features = weight.shape
    assert in_features % NVFP4_BLOCK_SIZE == 0

    weight_f32 = weight.float()

    # Compute per-expert global scale: maps max abs value to FP4_MAX * FP8_MAX
    # global_scale = (FP4_MAX * FP8_MAX) / amax
    amax = weight_f32.abs().reshape(E, -1).amax(dim=-1).clamp(min=1e-12)
    global_scales = (FP4_MAX * FP8_MAX) / amax  # [E]

    # Reshape into blocks of 16
    num_blocks = in_features // NVFP4_BLOCK_SIZE
    blocked = weight_f32.reshape(E, out_features, num_blocks, NVFP4_BLOCK_SIZE)

    # Per-block scale: block_scale = global_scale * (block_max / FP4_MAX)
    block_max = blocked.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    gs = global_scales.reshape(E, 1, 1, 1)
    raw_block_scale = gs * (block_max / FP4_MAX)
    raw_block_scale = raw_block_scale.clamp(-FP8_MAX, FP8_MAX)

    # Cast to FP8-E4M3 and back for rounding
    block_scale_fp8 = raw_block_scale.to(torch.float8_e4m3fn)
    block_scale_f32 = block_scale_fp8.to(torch.float32)

    # Compute output scale: global_scale / block_scale
    output_scale = gs / block_scale_f32.clamp(min=1e-12)

    # Scale input and clamp to FP4 range
    scaled = blocked * output_scale
    clamped = scaled.clamp(-FP4_MAX, FP4_MAX)

    # Round to nearest E2M1 value
    rounded = _round_to_fp4_e2m1(clamped)

    # Encode FP4 values to 4-bit indices
    indices = _fp4_to_index(rounded)  # [E, out, num_blocks, 16]

    # Pack 2 FP4 values per byte: low nibble = even index, high nibble = odd index
    indices = indices.reshape(E, out_features, num_blocks, NVFP4_BLOCK_SIZE // 2, 2)
    packed = (indices[..., 0] & 0x0F) | ((indices[..., 1] & 0x0F) << 4)
    packed = packed.reshape(E, out_features, in_features // 2).to(torch.uint8)

    block_scales_out = block_scale_fp8.squeeze(-1)  # [E, out, num_blocks]

    return packed, block_scales_out, global_scales


def _round_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round float values to nearest FP4 E2M1 representable values."""
    sign = torch.sign(x)
    abs_x = x.abs()
    # FP4 E2M1 representable magnitudes: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    # Boundaries (midpoints): 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
    result = torch.zeros_like(abs_x)
    result[abs_x >= 0.25] = 0.5
    result[abs_x >= 0.75] = 1.0
    result[abs_x >= 1.25] = 1.5
    result[abs_x >= 1.75] = 2.0
    result[abs_x >= 2.5] = 3.0
    result[abs_x >= 3.5] = 4.0
    result[abs_x >= 5.0] = 6.0
    return result * sign


def _fp4_to_index(x: torch.Tensor) -> torch.Tensor:
    """Convert FP4 E2M1 float values to 4-bit index (0-15)."""
    sign = (x < 0).to(torch.int64) * 8  # sign bit = bit 3
    abs_x = x.abs()
    # Map magnitude to 3-bit index
    magnitude = torch.zeros_like(abs_x, dtype=torch.int64)
    magnitude[abs_x >= 0.25] = 1   # 0.5
    magnitude[abs_x >= 0.75] = 2   # 1.0
    magnitude[abs_x >= 1.25] = 3   # 1.5
    magnitude[abs_x >= 1.75] = 4   # 2.0
    magnitude[abs_x >= 2.5] = 5    # 3.0
    magnitude[abs_x >= 3.5] = 6    # 4.0
    magnitude[abs_x >= 5.0] = 7    # 6.0
    return sign + magnitude


def convert_checkpoint(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Load config
    config_path = os.path.join(input_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    num_experts = config["num_local_experts"]
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    print(f"Model: {num_experts} experts, hidden={hidden_size}, intermediate={intermediate_size}")

    modules_to_not_convert = config.get("quantization_config", {}).get(
        "modules_to_not_convert", []
    )
    print(f"Modules NOT converted (kept bf16): {modules_to_not_convert}")

    # Find safetensor files
    shard_files = sorted(glob.glob(os.path.join(input_dir, "model*.safetensors")))
    print(f"Found {len(shard_files)} safetensor shards")

    # Process each shard
    shard_idx = 0
    weight_map = {}

    for shard_file in shard_files:
        print(f"\nProcessing {os.path.basename(shard_file)}...")
        output_tensors = {}

        with safe_open(shard_file, framework="pt", device="cpu") as st:
            keys = sorted(st.keys())
            for key in keys:
                tensor = st.get_tensor(key)

                if "gate_up_proj_blocks" in key:
                    scale_key = key.replace("gate_up_proj_blocks", "gate_up_proj_scales")
                    scale_tensor = st.get_tensor(scale_key)
                    _convert_moe_weight(
                        output_tensors, key, tensor, scale_tensor,
                        "gate_up_proj", num_experts, is_gate_up=True,
                    )
                elif "down_proj_blocks" in key:
                    scale_key = key.replace("down_proj_blocks", "down_proj_scales")
                    scale_tensor = st.get_tensor(scale_key)
                    _convert_moe_weight(
                        output_tensors, key, tensor, scale_tensor,
                        "down_proj", num_experts, is_gate_up=False,
                    )
                elif "gate_up_proj_scales" in key or "down_proj_scales" in key:
                    # Already handled with the blocks
                    continue
                else:
                    # Non-MoE weights: keep as-is
                    output_tensors[key] = tensor

        # Save output shard
        out_name = f"model-{shard_idx:05d}-of-{len(shard_files):05d}.safetensors"
        out_path = os.path.join(output_dir, out_name)
        print(f"  Saving {out_name} with {len(output_tensors)} tensors...")
        save_file(output_tensors, out_path)

        for k in output_tensors:
            weight_map[k] = out_name

        shard_idx += 1

    # Save model index
    index = {
        "metadata": {"total_size": sum(
            t.numel() * t.element_size()
            for f in glob.glob(os.path.join(output_dir, "model*.safetensors"))
            for t in [torch.zeros(1)]  # placeholder
        )},
        "weight_map": weight_map,
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # Save updated config
    new_config = dict(config)
    new_config["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "group_size": 16,
        "kv_cache_quant_algo": None,
        "exclude_modules": modules_to_not_convert,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(new_config, f, indent=2)

    # Save hf_quant_config.json (for ModelOpt detection)
    hf_quant_config = {
        "quant_method": "modelopt",
        "quantization": {
            "quant_algo": "NVFP4",
            "group_size": 16,
            "kv_cache_quant_algo": None,
            "exclude_modules": modules_to_not_convert,
        }
    }
    with open(os.path.join(output_dir, "hf_quant_config.json"), "w") as f:
        json.dump(hf_quant_config, f, indent=2)

    # Copy tokenizer files
    for fname in os.listdir(input_dir):
        if fname.startswith("tokenizer") or fname in (
            "special_tokens_map.json", "generation_config.json",
        ):
            src = os.path.join(input_dir, fname)
            dst = os.path.join(output_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"Copied {fname}")

    print(f"\nDone! NVFP4 checkpoint saved to {output_dir}")


def _convert_moe_weight(
    output_tensors: dict,
    key: str,
    blocks: torch.Tensor,
    scales: torch.Tensor,
    proj_name: str,
    num_experts: int,
    is_gate_up: bool,
) -> None:
    """Dequantize MXFP4 and re-quantize to NVFP4."""
    print(f"  Converting {key}: {list(blocks.shape)} -> NVFP4")

    # Dequantize MXFP4 -> bf16
    bf16_weight = dequantize_mxfp4(blocks, scales)
    print(f"    Dequantized to bf16: {list(bf16_weight.shape)}, "
          f"range=[{bf16_weight.float().min():.4f}, {bf16_weight.float().max():.4f}]")

    # Quantize bf16 -> NVFP4
    packed, block_scales, global_scales = quantize_nvfp4(bf16_weight)
    print(f"    NVFP4 packed: {list(packed.shape)}, "
          f"block_scales: {list(block_scales.shape)}, "
          f"global_scales: {list(global_scales.shape)}")

    # Build output key prefix (replace blocks suffix with quark-style names)
    # e.g. "model.layers.0.mlp.experts.gate_up_proj_blocks"
    #   -> "model.layers.0.mlp.experts.gate_up_proj.weight"
    base_key = key.replace(f"{proj_name}_blocks", proj_name)

    output_tensors[f"{base_key}.weight"] = packed
    output_tensors[f"{base_key}.weight_scale"] = block_scales
    if is_gate_up:
        # gate_up has 2 shards (gate + up), so global_scale is [E, 2]
        # Compute separate global scales for gate and up halves
        E = bf16_weight.shape[0]
        out_half = bf16_weight.shape[1] // 2
        gate_amax = bf16_weight[:, :out_half, :].float().abs().reshape(E, -1).amax(dim=-1).clamp(min=1e-12)
        up_amax = bf16_weight[:, out_half:, :].float().abs().reshape(E, -1).amax(dim=-1).clamp(min=1e-12)
        gate_gs = (FP4_MAX * FP8_MAX) / gate_amax
        up_gs = (FP4_MAX * FP8_MAX) / up_amax
        gs_2 = torch.stack([gate_gs, up_gs], dim=-1)  # [E, 2]
        input_scale = torch.ones(num_experts, 2, dtype=torch.float32)
    else:
        gs_2 = global_scales  # [E]
        input_scale = torch.ones(num_experts, dtype=torch.float32)

    output_tensors[f"{base_key}.weight_scale_2"] = gs_2
    output_tensors[f"{base_key}.input_scale"] = input_scale

    # Verify round-trip accuracy
    _verify_roundtrip(bf16_weight, packed, block_scales, global_scales)


def _verify_roundtrip(
    original: torch.Tensor,
    packed: torch.Tensor,
    block_scales: torch.Tensor,
    global_scales: torch.Tensor,
) -> None:
    """Verify NVFP4 quantization by dequantizing and comparing."""
    E, out_features, half_in = packed.shape
    in_features = half_in * 2

    # Unpack
    low = (packed & 0x0F).to(torch.int64)
    high = ((packed >> 4) & 0x0F).to(torch.int64)
    unpacked = torch.stack([low, high], dim=-1).reshape(E, out_features, in_features)
    fp4_values = FP4_E2M1_LUT[unpacked]

    # Dequant
    num_blocks = in_features // NVFP4_BLOCK_SIZE
    fp4_blocked = fp4_values.reshape(E, out_features, num_blocks, NVFP4_BLOCK_SIZE)
    bs = block_scales.to(torch.float32).unsqueeze(-1)
    gs = global_scales.reshape(E, 1, 1, 1)
    dequant = (fp4_blocked * bs / gs).reshape(E, out_features, in_features)

    # Compare
    orig_f32 = original.float()
    error = (dequant - orig_f32).abs()
    rel_error = error / (orig_f32.abs().clamp(min=1e-8))
    print(f"    Round-trip: max_abs_error={error.max():.6f}, "
          f"mean_rel_error={rel_error.mean():.4%}")


def main():
    parser = argparse.ArgumentParser(description="Convert GPT-OSS MXFP4 to NVFP4")
    parser.add_argument("--input-dir", required=True, help="Path to MXFP4 HF checkpoint")
    parser.add_argument("--output-dir", required=True, help="Path to save NVFP4 checkpoint")
    args = parser.parse_args()
    convert_checkpoint(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
