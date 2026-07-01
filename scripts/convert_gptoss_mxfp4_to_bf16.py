"""
Dequantize GPT-OSS-20B from MXFP4 to bf16 HF checkpoint.

MXFP4 (OCP MX): E2M1 FP4 values with E8M0 block scales, block_size=32.
Only MoE expert weights are quantized; attention, embeddings, router are already bf16.

This produces a standard bf16 HF checkpoint that can be loaded by
AutoModelForCausalLM and then quantized with ModelOpt.

Usage:
    # download openai/gpt-oss-20b from HF and write to /tmp/gpt-oss-20b-bf16
    python convert_gptoss_mxfp4_to_bf16.py

    # or specify an existing MXFP4 checkpoint and/or output dir
    python convert_gptoss_mxfp4_to_bf16.py --input-dir <mxfp4_ckpt> --output-dir <bf16_out>
"""

import argparse
import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


FP4_E2M1_LUT = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def dequantize_mxfp4(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 packed data to bf16.

    Args:
        blocks: [E, out_features, num_blocks, 16] uint8
        scales: [E, out_features, num_blocks] uint8 (E8M0)
    Returns:
        [E, out_features, in_features] bf16
    """
    E, out_features, num_blocks, bytes_per_block = blocks.shape
    assert bytes_per_block == 16

    low_nibble = (blocks & 0x0F).to(torch.int64)
    high_nibble = ((blocks >> 4) & 0x0F).to(torch.int64)
    unpacked = torch.stack([low_nibble, high_nibble], dim=-1)
    unpacked = unpacked.reshape(E, out_features, num_blocks, 32)

    fp4_values = FP4_E2M1_LUT[unpacked]

    e8m0_scale = torch.pow(2.0, scales.to(torch.float32) - 127.0).unsqueeze(-1)

    dequantized = fp4_values * e8m0_scale
    in_features = num_blocks * 32
    return dequantized.reshape(E, out_features, in_features).to(torch.bfloat16)


def convert_checkpoint(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(input_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    print(f"Model: {config['num_local_experts']} experts, "
          f"hidden={config['hidden_size']}, intermediate={config['intermediate_size']}")

    shard_files = sorted(glob.glob(os.path.join(input_dir, "model*.safetensors")))
    print(f"Found {len(shard_files)} safetensor shards")

    weight_map = {}

    for shard_idx, shard_file in enumerate(shard_files):
        print(f"\nProcessing {os.path.basename(shard_file)}...")
        output_tensors = {}

        with safe_open(shard_file, framework="pt", device="cpu") as st:
            keys = sorted(st.keys())
            for key in keys:
                tensor = st.get_tensor(key)

                if "gate_up_proj_blocks" in key:
                    scale_key = key.replace("gate_up_proj_blocks", "gate_up_proj_scales")
                    scale_tensor = st.get_tensor(scale_key)
                    bf16_weight = dequantize_mxfp4(tensor, scale_tensor)
                    # MXFP4 stores [E, 2*I, H], HF expects [E, H, 2*I]
                    bf16_weight = bf16_weight.transpose(1, 2).contiguous()
                    new_key = key.replace("gate_up_proj_blocks", "gate_up_proj")
                    output_tensors[new_key] = bf16_weight
                    print(f"  {key} -> {new_key}: {list(bf16_weight.shape)} bf16")
                elif "down_proj_blocks" in key:
                    scale_key = key.replace("down_proj_blocks", "down_proj_scales")
                    scale_tensor = st.get_tensor(scale_key)
                    bf16_weight = dequantize_mxfp4(tensor, scale_tensor)
                    # MXFP4 stores [E, H, I], HF expects [E, I, H]
                    bf16_weight = bf16_weight.transpose(1, 2).contiguous()
                    new_key = key.replace("down_proj_blocks", "down_proj")
                    output_tensors[new_key] = bf16_weight
                    print(f"  {key} -> {new_key}: {list(bf16_weight.shape)} bf16")
                elif "scales" in key and ("gate_up_proj" in key or "down_proj" in key):
                    continue
                else:
                    output_tensors[key] = tensor

        out_name = f"model-{shard_idx + 1:05d}-of-{len(shard_files):05d}.safetensors"
        out_path = os.path.join(output_dir, out_name)
        print(f"  Saving {out_name} with {len(output_tensors)} tensors...")
        save_file(output_tensors, out_path)

        for k in output_tensors:
            weight_map[k] = out_name

    # Compute total size
    total_size = 0
    for f in glob.glob(os.path.join(output_dir, "model*.safetensors")):
        total_size += os.path.getsize(f)

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # Save config WITHOUT quantization_config
    new_config = {k: v for k, v in config.items() if k != "quantization_config"}
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(new_config, f, indent=2)

    # Copy tokenizer and other files
    for fname in os.listdir(input_dir):
        if fname.startswith("tokenizer") or fname in (
            "special_tokens_map.json", "generation_config.json",
        ):
            src = os.path.join(input_dir, fname)
            dst = os.path.join(output_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"Copied {fname}")

    print(f"\nDone! bf16 checkpoint saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=None,
        help="MXFP4 checkpoint dir. If omitted, openai/gpt-oss-20b is downloaded from HF.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/gpt-oss-20b-bf16",
        help="Where to write the bf16 checkpoint (default: /tmp/gpt-oss-20b-bf16).",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    if input_dir is None:
        from huggingface_hub import snapshot_download

        print("Downloading openai/gpt-oss-20b from Hugging Face...")
        input_dir = snapshot_download("openai/gpt-oss-20b")

    convert_checkpoint(input_dir, args.output_dir)


if __name__ == "__main__":
    main()
