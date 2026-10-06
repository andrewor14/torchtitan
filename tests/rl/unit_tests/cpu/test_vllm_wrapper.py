# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import spmd_types as spmd

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.rl.model.vllm_wrapper import _replace_vllm_layer_configs


def test_vllm_replacement_declares_attention_scale_sharding() -> None:
    """Preserve attention layouts and replicate vLLM-only scale buffers."""
    model_config = build_model_config("debugmodel", attn_backend="flex")
    model_config.set_sharding_(
        ParallelismConfig(tensor_parallel_degree=2, enable_sequence_parallel=True)
    )
    model_config.layers = [
        layer for layer in model_config.layers if layer.attention is not None
    ]

    vllm_config = _replace_vllm_layer_configs(model_config)
    replicated_dense = dense_param_placement(tp=spmd.R)

    for model_layer, vllm_layer in zip(
        model_config.layers, vllm_config.layers, strict=True
    ):
        assert model_layer.attention is not None
        assert vllm_layer.attention is not None
        model_sharding = model_layer.attention.inner_attention.sharding_config
        vllm_attention_config = vllm_layer.attention.inner_attention
        vllm_sharding = vllm_attention_config.sharding_config
        assert model_sharding is not None
        assert vllm_sharding is not None
        assert vllm_sharding.in_src_shardings is model_sharding.in_src_shardings
        assert vllm_sharding.in_dst_shardings is model_sharding.in_dst_shardings
        assert vllm_sharding.out_src_shardings is model_sharding.out_src_shardings
        assert vllm_sharding.out_dst_shardings is model_sharding.out_dst_shardings
        assert vllm_sharding.local_spmd is model_sharding.local_spmd
        for name, layout in model_sharding.state_shardings.items():
            assert vllm_sharding.state_shardings[name] is layout
        vllm_attn_sharding = vllm_attention_config.vllm_attn_sharding_config
        for name in ("_k_scale", "_prob_scale", "_q_scale", "_v_scale"):
            assert vllm_attn_sharding.state_shardings[name] == replicated_dense
