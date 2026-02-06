# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from importlib.util import find_spec
from typing import List

import torch.nn as nn
from torchtitan.components.quantization import QuantizationConverter

from torchtitan.config.job_config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import register_model_converter
from torchtitan.tools.logging import logger

from .utils import module_filter_fn


class QATLinearConverter(QuantizationConverter):
    """Converts the linear layers of `model` to `FakeQuantizedLinear`."""

    filter_fqns: List[str]

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        super().__init__(job_config, parallel_dims)
        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use QAT."
            )

        # Configure QAT
        from torchao.quantization import (
            Int4WeightOnlyConfig,
        )

        qat_job_config = job_config.quantize.linear.qat
        if qat_job_config.recipe_name == "int4":
            self.base_config = Int4WeightOnlyConfig()
        else:
            raise ValueError(
                f"Unknown QAT recipe name {qat_job_config.recipe_name}, "
                "only 'int4' is supported at the moment"
            )
        self.filter_fqns = qat_job_config.filter_fqns
        self.enabled = True
        logger.info(f"QAT active with recipe {qat_job_config.recipe_name}")

    def convert(self, model: nn.Module):
        """
        Converts the linear layers of `model` to `FakeQuantizedLinear`.
        This will mutate the model inplace.
        """
        if not self.enabled:
            return

        from torchao.core.config import AOBaseConfig
        from torchao.quantization import quantize_
        from torchao.quantization.qat import QATConfig

        assert isinstance(self.config, AOBaseConfig)
        quantize_(
            model,
            config=QATConfig(self.base_config, step="prepare"),
            filter_fn=partial(module_filter_fn, filter_fqns=self.filter_fqns),
        )
        logger.info("Swapped to FakeQuantized layers for QAT")

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        """
        QAT doesn't require any post-optimizer hooks at the moment.
        """
        return


register_model_converter(QATLinearConverter, "quantize.linear.qat")
