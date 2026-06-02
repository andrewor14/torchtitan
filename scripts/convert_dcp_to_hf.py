"""Convert a DCP checkpoint to HF safetensors format."""
import argparse
import json
import shutil
import torch
import torch.distributed.checkpoint as dcp
import importlib
from pathlib import Path
from safetensors.torch import save_file
from torchtitan.components.checkpoint import ModelWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dcp_path", type=str, help="Path to DCP checkpoint dir")
    parser.add_argument("output_path", type=str, help="Output HF checkpoint dir")
    parser.add_argument("--hf_src", type=str,
                        default="/home/andrewor/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
                        help="Original HF model dir (for config/tokenizer)")
    parser.add_argument("--model_name", type=str, default="qwen3")
    parser.add_argument("--model_flavor", type=str, default="30B-A3B")
    parser.add_argument("--hf_assets_path", type=str, default="./assets/hf/Qwen3-30B-A3B")
    args = parser.parse_args()

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_module = importlib.import_module(f"torchtitan.models.{args.model_name}")
    model_spec = model_module.model_registry(args.model_flavor)
    model_config = model_spec.model

    with torch.device("cpu"):
        model = model_config.build()
    model = ModelWrapper(model)

    sd = model.state_dict()
    dcp.load(sd, checkpoint_id=args.dcp_path)

    sd_adapter = model_spec.state_dict_adapter(model_config, args.hf_assets_path)
    hf_sd = sd_adapter.to_hf(sd)

    for k in hf_sd:
        if hf_sd[k].dtype == torch.float32:
            hf_sd[k] = hf_sd[k].to(torch.bfloat16)

    print(f"Saving {len(hf_sd)} tensors to {output_dir}...")
    save_file(hf_sd, output_dir / "model.safetensors")

    weight_map = {k: "model.safetensors" for k in hf_sd}
    index = {
        "metadata": {"total_size": sum(v.numel() * v.element_size() for v in hf_sd.values())},
        "weight_map": weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    hf_src = Path(args.hf_src)
    for fname in ["config.json", "generation_config.json", "tokenizer.json",
                   "tokenizer_config.json", "merges.txt", "vocab.json",
                   "special_tokens_map.json"]:
        src = hf_src / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)

    print("Done!")


if __name__ == "__main__":
    main()
