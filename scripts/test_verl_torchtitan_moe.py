"""
Minimal test: verify verl's torchtitan engine works with Qwen3-30B-A3B MoE.

Tests:
1. Model resolution (HF config -> torchtitan name/flavor)
2. Engine initialization (model build + HF weight loading)
3. Forward pass (with dummy data)
4. Weight sync (get_per_tensor_param produces all expert weights)

Run with: CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --rdzv_backend c10d \
          --rdzv_endpoint="localhost:0" scripts/test_verl_torchtitan_moe.py
"""
import os
import torch
import torch.distributed as dist

from transformers import AutoConfig
from verl.workers.engine.torchtitan.utils import derive_torchtitan_name_and_flavor
from verl.workers.engine.torchtitan.transformer_impl import TorchTitanEngine
from verl.workers.config.engine import TorchtitanEngineConfig
from verl.workers.config.optimizer import TorchtitanOptimizerConfig
from verl.workers.config import HFModelConfig
from verl.trainer.config import CheckpointConfig


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    hf_model_path = os.environ.get(
        "MODEL_PATH",
        "/home/andrewor/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
    )

    # 1. Model resolution
    print("=== Step 1: Model resolution ===")
    hf_config = AutoConfig.from_pretrained(hf_model_path)
    name, flavor = derive_torchtitan_name_and_flavor(hf_config)
    print(f"  HF model_type: {hf_config.model_type}")
    print(f"  torchtitan: name={name}, flavor={flavor}")
    assert name == "qwen3" and flavor == "30B-A3B", f"Wrong resolution: {name}/{flavor}"

    # 2. Engine initialization
    print("\n=== Step 2: Engine initialization ===")
    model_config = HFModelConfig(
        path=hf_model_path,
        enable_gradient_checkpointing=False,
        use_remove_padding=False,
    )
    model_config.hf_config = hf_config

    engine_config = TorchtitanEngineConfig(
        data_parallel_shard_size=-1,
        data_parallel_replicate_size=1,
        tensor_parallel_size=1,
        expert_parallel_size=1,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        use_torch_compile=False,
        forward_only=False,
        dtype="bfloat16",
        attn_type="sdpa",
    )

    optim_config = TorchtitanOptimizerConfig(
        lr=1e-6,
        total_training_steps=10,
    )

    checkpoint_config = CheckpointConfig()

    engine = TorchTitanEngine(
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optim_config,
        checkpoint_config=checkpoint_config,
    )
    engine.initialize()
    print(f"  Model loaded successfully")
    print(f"  Module type: {type(engine.module[0])}")

    # Check model has MoE layers
    model = engine.module[0]
    sd = model.state_dict()
    expert_keys = [k for k in sd if "moe" in k or "expert" in k]
    print(f"  Expert/MoE state dict keys: {len(expert_keys)}")
    assert len(expert_keys) > 0, "No MoE keys found in model state dict!"

    # 3. Forward pass
    print("\n=== Step 3: Forward pass ===")
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, hf_config.vocab_size, (batch_size, seq_len), device=device)

    model.train()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(input_ids)
    print(f"  Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, hf_config.vocab_size), f"Wrong shape: {output.shape}"

    # 4. Weight sync (get_per_tensor_param)
    print("\n=== Step 4: Weight sync (get_per_tensor_param) ===")
    per_tensor_param, peft_config = engine.get_per_tensor_param()

    hf_keys = []
    expert_count = 0
    for name, tensor in per_tensor_param:
        hf_keys.append(name)
        if "expert" in name:
            expert_count += 1

    print(f"  Total HF keys yielded: {len(hf_keys)}")
    print(f"  Expert keys: {expert_count}")
    assert expert_count == 18432, f"Expected 18432 expert keys, got {expert_count}"
    print(f"  Weight sync: OK")

    print("\n=== ALL TESTS PASSED ===")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
