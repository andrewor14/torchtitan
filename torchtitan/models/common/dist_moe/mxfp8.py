# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 prepared-weight lifecycle for Dist-MoE grouped projections."""

from __future__ import annotations

from dataclasses import dataclass, field

import spmd_types as spmd
import torch

from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.quantization._fsdp_tensor import (
    _ShardedFSDPTensor,
    _UnshardedFSDPTensor,
)

from . import _dist_moe as dist_moe
from .routed_experts import DistMoeRoutedExperts


__all__ = ["MXFP8DistMoeRoutedExperts"]


@dataclass(frozen=True, slots=True)
class _DistMoeMXFP8Operands:
    """Independent MXFP8 tensors owned by one FSDP unshard lifetime.

    Forward-only (``inference``) operands have no DGRAD scales. Every field is
    FSDP-managed storage and cannot be ``None``, so ``dgrad_scale`` is then an
    empty tensor.
    """

    qdata: torch.Tensor
    fprop_scale: torch.Tensor
    dgrad_scale: torch.Tensor

    def prepared(
        self, source: torch.Tensor, *, fuse_swiglu: bool = False
    ) -> dist_moe.PreparedWeight:
        """Return the annex facade consumed by one Dist-MoE invocation.

        ``fuse_swiglu`` marks W13 operands whose gate/up rows are interleaved
        for the annex's fused SwiGLU Mega inference kernel.
        """
        forward_only = self.dgrad_scale.numel() == 0
        return dist_moe.PreparedWeight._create(
            source=source,
            format=dist_moe.BlockScaledFormat.MXFP8_E4M3,
            fprop_data=self.qdata,
            fprop_scale=self.fprop_scale,
            dgrad_data=None if forward_only else self.qdata,
            dgrad_scale=None if forward_only else self.dgrad_scale,
            interleaved_fc13=fuse_swiglu,
        )


def _prepare_mxfp8_weight(
    weight_EOI: torch.Tensor,
    out: _DistMoeMXFP8Operands | None = None,
    *,
    inference: bool,
    fuse_swiglu: bool = False,
) -> _DistMoeMXFP8Operands:
    """Allocate or refill the annex's grouped MXFP8 weight operands.

    See ``_DistMoeMXFP8ShardedTensor`` for the operand formats.
    ``fuse_swiglu`` requires ``inference``.
    """
    if fuse_swiglu and not inference:
        raise ValueError("fuse_swiglu requires inference=True")
    prepared = dist_moe.prepare_block_scaled_weight(
        weight_EOI,
        dist_moe.BlockScaledConfig(pipeline="mega" if fuse_swiglu else "staged"),
        inference=inference,
        fuse_swiglu=fuse_swiglu,
    )
    if not inference:
        if prepared.dgrad_data is not prepared.fprop_data:
            raise RuntimeError("MXFP8 Dist-MoE FPROP and DGRAD must share qdata")
        if prepared.dgrad_scale is None:
            raise RuntimeError(
                "MXFP8 Dist-MoE preparation returned incomplete operands"
            )
    operands = _DistMoeMXFP8Operands(
        qdata=prepared.fprop_data,
        fprop_scale=prepared.fprop_scale,
        dgrad_scale=(
            prepared.fprop_scale.new_empty((0,)) if inference else prepared.dgrad_scale
        ),
    )
    if out is None:
        return operands
    out.qdata.copy_(operands.qdata)
    out.fprop_scale.copy_(operands.fprop_scale)
    out.dgrad_scale.copy_(operands.dgrad_scale)
    return out


class _DistMoeMXFP8ShardedTensor(_ShardedFSDPTensor):
    """Abstract persistent grouped expert weight that prepares MXFP8 operands.

    Subclasses implement ``_build_operands`` for W13 or W2.

    When ``inference=False``, operands use grouped 32x32 quantization with
    FPROP and DGRAD scales, usable by both training and inference.

    When ``inference=True``, operands are forward-only, with 1x32 quantization
    and no DGRAD scales, for the annex's fused SwiGLU Mega kernel. Only W13 is
    prepared for fused SwiGLU (interleaved gate/up rows); W2 is not.
    """

    def __init__(
        self, tensor: torch.Tensor, *, inference: bool = False, **logical_metadata
    ) -> None:
        super().__init__(tensor, **logical_metadata)
        self.inference = inference

    def __tensor_flatten__(self):
        return ["_tensor"], (self.dtype, self.inference)

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, metadata, outer_size, outer_stride):
        del outer_size, outer_stride
        _, inference = metadata
        return cls(inner_tensors["_tensor"], inference=inference)

    def _rewrap(self, tensor: torch.Tensor) -> _DistMoeMXFP8ShardedTensor:
        return type(self)(tensor, inference=self.inference)


class _DistMoeW13ShardedTensor(_DistMoeMXFP8ShardedTensor):
    """Persistent W13 parameter that prepares grouped MXFP8 compute operands."""

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _DistMoeMXFP8Operands | None = None,
    ) -> _DistMoeMXFP8Operands:
        return _prepare_mxfp8_weight(
            logical_tensor.flatten(1, 2),
            out,
            inference=self.inference,
            fuse_swiglu=self.inference,
        )


class _DistMoeW2ShardedTensor(_DistMoeMXFP8ShardedTensor):
    """Persistent W2 parameter that prepares grouped MXFP8 compute operands."""

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _DistMoeMXFP8Operands | None = None,
    ) -> _DistMoeMXFP8Operands:
        return _prepare_mxfp8_weight(logical_tensor, out, inference=self.inference)


def _dynamic_prepared_weight(
    logical_weight: torch.Tensor,
    *,
    gate_up: bool,
    inference: bool,
) -> dist_moe.PreparedWeight:
    """Prepare a weight when FSDP does not own its unshard lifetime."""
    fuse_swiglu = inference and gate_up
    source = logical_weight.flatten(1, 2) if gate_up else logical_weight
    storage = (
        logical_weight._tensor
        if isinstance(logical_weight, _ShardedFSDPTensor)
        else logical_weight
    )
    compute_weight = storage.flatten(1, 2) if gate_up else storage
    with torch.no_grad():
        operands = _prepare_mxfp8_weight(
            compute_weight, inference=inference, fuse_swiglu=fuse_swiglu
        )
    return operands.prepared(source, fuse_swiglu=fuse_swiglu)


class MXFP8DistMoeRoutedExperts(DistMoeRoutedExperts):
    """Dist-MoE routed experts using native asynchronous MXFP8 kernels.

    The module preserves the inherited W13/W2 parameter and checkpoint layout,
    while TorchTitan's shared FSDP tensor lifecycle prepares the qdata and scale
    layouts required by Dist-MoE for each unshard lifetime. Without FSDP, the
    same layouts are prepared directly from the live parameters before use.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DistMoeRoutedExperts.Config):
        """Configure native MXFP8 execution for Dist-MoE routed experts.

        Args:
            block_scaled_config: Annex MXFP8 kernel policy. Its ``pipeline``
                selects separate staged kernels or the fused Mega pipeline;
                ``fast_math`` enables approximate sigmoid math in fused SwiGLU;
                and ``kernel_config`` is an expert-only CuTe tuning override.
            inference: Prepare forward-only weights for the annex's fused
                SwiGLU Mega inference kernel (see
                ``_DistMoeMXFP8ShardedTensor``). Requires ``pipeline="mega"``
                and an inference runtime; the modules cannot run backward.
                Parameters and checkpoints keep the canonical layout.
        """

        block_scaled_config: dist_moe.BlockScaledConfig = field(
            default_factory=lambda: dist_moe.BlockScaledConfig()
        )
        inference: bool = False

        def __post_init__(self) -> None:
            """Validate the common expert and MXFP8-specific policies."""
            DistMoeRoutedExperts.Config.__post_init__(self)
            if (
                self.block_scaled_config.format
                is not dist_moe.BlockScaledFormat.MXFP8_E4M3
            ):
                raise ValueError(
                    "MXFP8DistMoeRoutedExperts requires the MXFP8_E4M3 format"
                )
            if self.bf16_grouped_gemm_preset is not None:
                raise ValueError(
                    "bf16_grouped_gemm_preset is not used by MXFP8 Dist-MoE"
                )
            if self.inference and self.block_scaled_config.pipeline != "mega":
                raise ValueError(
                    "inference requires block_scaled_config.pipeline='mega'"
                )

    def __init__(self, config: Config):
        super().__init__(config)
        self.block_scaled_config = config.block_scaled_config
        self.inference = config.inference
        self.w13.weight = torch.nn.Parameter(
            _DistMoeW13ShardedTensor(self.w13.weight.data, inference=self.inference),
            requires_grad=self.w13.weight.requires_grad,
        )
        self.w2.weight = torch.nn.Parameter(
            _DistMoeW2ShardedTensor(self.w2.weight.data, inference=self.inference),
            requires_grad=self.w2.weight.requires_grad,
        )

    def _parallelize(self, parallelism_context: ParallelismContext) -> None:
        # spmd_types returns a plain tensor when EP/TP shards a weight. Restore
        # the FSDP extension wrappers before fully_shard() consumes them.
        super()._parallelize(parallelism_context)
        for proj, wrapper_cls in (
            (self.w13, _DistMoeW13ShardedTensor),
            (self.w2, _DistMoeW2ShardedTensor),
        ):
            if isinstance(proj.weight, wrapper_cls):
                continue
            distributed_weight = proj.weight
            wrapped_weight = torch.nn.Parameter(
                wrapper_cls(distributed_weight.data, inference=self.inference),
                requires_grad=distributed_weight.requires_grad,
            )
            spmd.assert_type_like(wrapped_weight, distributed_weight)
            proj.weight = wrapped_weight

    def _weight_operands(
        self,
    ) -> tuple[
        torch.Tensor | dist_moe.PreparedWeight,
        torch.Tensor | dist_moe.PreparedWeight,
    ]:
        """Return prepared W13 and W2 operands for this unshard lifetime."""
        w13_E2FD = self.w13.weight
        w2_EDF = self.w2.weight
        w13_EFD = w13_E2FD.flatten(1, 2)
        w13_operand = (
            w13_E2FD.operands.prepared(w13_EFD, fuse_swiglu=self.inference)
            if isinstance(w13_E2FD, _UnshardedFSDPTensor)
            else _dynamic_prepared_weight(
                w13_E2FD,
                gate_up=True,
                inference=self.inference,
            )
        )
        w2_operand = (
            w2_EDF.operands.prepared(w2_EDF)
            if isinstance(w2_EDF, _UnshardedFSDPTensor)
            else _dynamic_prepared_weight(
                w2_EDF, gate_up=False, inference=self.inference
            )
        )
        return w13_operand, w2_operand
