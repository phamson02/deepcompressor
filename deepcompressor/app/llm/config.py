# -*- coding: utf-8 -*-
"""Configurations for evaluating a large language model."""

import os
import random
from dataclasses import dataclass, field

import numpy as np
import omniconfig
import torch
from omniconfig import ConfigParser, configclass

from deepcompressor.data.utils import ScaleUtils
from deepcompressor.utils.config.output import OutputConfig

from .cache.config import LlmCacheConfig, LlmQuantCacheConfig
from .eval.config import LlmEvalConfig
from .model.config import LlmModelConfig
from .quant.config import LlmQuantConfig

__all__ = [
    "LlmPtqRunConfig",
    "LlmCacheConfig",
    "LlmQuantCacheConfig",
    "LlmEvalConfig",
    "LlmModelConfig",
    "LlmQuantConfig",
]


@configclass
@dataclass
class LlmPtqRunConfig:
    """Top-level config of post-training quantization for a large language model.

    Args:
        cache (`LlmCacheConfig`):
            Large language model quantization cache path configuration.
        output (`OutputConfig`):
            Output directory configuration.
        model (`LlmModelConfig`):
            Large language model configuration.
        eval (`LlmEvalConfig`):
            Large language model evaluation configuration.
        quant (`LlmQuantConfig`):
            Large language model quantization configuration.
        seed (`int`, *optional*, defaults to `12345`):
            Random seed.
        skip_eval (`bool`, *optional*, defaults to `False`):
            Whether to skip evaluation.
        load_model (`str`, *optional*, defaults to `""`):
            Directory path to load the model checkpoint.
        save_model (`str`, *optional*, defaults to `""`):
            Directory path to save the model checkpoint.
        copy_on_save (`bool`, *optional*, defaults to `False`):
            Whether to copy the quantization cache on save.
    """

    cache: LlmCacheConfig
    output: OutputConfig
    model: LlmModelConfig
    eval: LlmEvalConfig
    quant: LlmQuantConfig = field(metadata={omniconfig.ARGPARSE_KWARGS: {"prefix": ""}})
    seed: int = 12345
    skip_eval: bool = False
    load_from: str = ""
    save_model: str = ""
    copy_on_save: bool = False
    # Single-layer ΔLoss scan settings
    delta_single_layer: bool = False
    delta_fields: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate",
            "down_proj",
            "moe_gate",
        ],
        metadata={omniconfig.ARGPARSE_KWARGS: {"nargs": "*", "type": str}},
    )
    # Metric to compare in delta mode: "ppl", "nll" (paired NLL), or "auto" (defaults to paired NLL)
    delta_metric: str = field(default="auto")
    # Compute activation metrics between original and quantized models
    act_metrics: bool = field(default=False, metadata={omniconfig.ARGPARSE_ARGS: ("--act-metrics",)})
    # Activation metrics KLD mode: softmax (per-row channel softmax), hist (JS over histograms), or hist_kl (KL(P||Q) over histograms)
    act_metrics_kld_mode: str = field(
        default="softmax",
        metadata={omniconfig.ARGPARSE_ARGS: ("--act-metrics-kld-mode",), "help": "softmax|hist|hist_kl"},
    )
    # Activation metrics histogram bins when kld-mode=hist (JS by default)
    act_metrics_bins: int = field(
        default=100,
        metadata={omniconfig.ARGPARSE_ARGS: ("--act-metrics-bins",), "help": "Histogram bins for hist mode"},
    )
    # Activation metrics aggregation: role (aggregate across modules) or module (report each module)
    act_metrics_aggregate: str = field(
        default="role",
        metadata={omniconfig.ARGPARSE_ARGS: ("--act-metrics-aggregate",), "help": "role|module"},
    )
    # Activation metrics source: calib (calibration loader) or eval (same dataset/windows as delta NLL)
    act_metrics_source: str = field(
        default="calib",
        metadata={omniconfig.ARGPARSE_ARGS: ("--act-metrics-source",), "help": "calib|eval"},
    )
    # Activation caching for ΔLoss scan: auto (use if present, build if missing), force (always build/use), skip (never cache)
    delta_act_cache: str = field(
        default="auto",
        metadata={omniconfig.ARGPARSE_ARGS: ("--delta-act-cache",), "help": "Activation cache mode: auto|force|skip"},
    )
    # Activation cache directory: default empty uses <run_dir>/delta/act_cache; use 'shm' to route to /dev/shm
    delta_act_cache_dir: str = field(
        default="",
        metadata={omniconfig.ARGPARSE_ARGS: ("--delta-act-cache-dir",), "help": "Dir for per-layer activation cache or 'shm'"},
    )
    # Delete activation cache directory at the end of ΔLoss scan
    delta_act_cache_cleanup: bool = field(
        default=True,
        metadata={omniconfig.ARGPARSE_ARGS: ("--delta-act-cache-cleanup",), "help": "Delete act_cache after scan"},
    )

    def __post_init__(self):  # noqa: C901
        # region set scale default dtype
        if self.quant.enabled_wgts:
            self.quant.wgts.scale_dtypes = tuple(
                ScaleUtils.infer_scale_dtypes(self.quant.wgts.scale_dtypes, default_dtype=self.model.dtype)
            )
        if self.quant.enabled_ipts:
            self.quant.ipts.scale_dtypes = tuple(
                ScaleUtils.infer_scale_dtypes(self.quant.ipts.scale_dtypes, default_dtype=self.model.dtype)
            )
        if self.quant.enabled_opts:
            self.quant.opts.scale_dtypes = tuple(
                ScaleUtils.infer_scale_dtypes(self.quant.opts.scale_dtypes, default_dtype=self.model.dtype)
            )
        # endregion
        # region set num_gpus and batch_size for auto parallelism of large models
        self.eval.num_gpus = min(torch.cuda.device_count(), self.eval.num_gpus)
        if self.model.size < 50:
            self.eval.batch_size = min(8, self.eval.batch_size)
        elif self.model.size < 100:
            self.eval.batch_size = min(4, self.eval.batch_size)
        else:
            self.eval.batch_size = min(1, self.eval.batch_size)
        # endregion
        if self.quant.is_enabled():
            if self.cache.path.is_all_empty():
                self.cache.dirpath = self.quant.generate_cache_dirpath(
                    root=self.cache.root, seed=self.seed, default_dtype=self.model.dtype
                )
                self.cache.path = self.cache.dirpath.clone().add_children(f"{self.model.name}.pt")
            else:
                self.cache.dirpath = self.cache.path.clone().to_dirpath()
        if self.output.dirname == "default":
            self.output.dirname = self.quant.generate_default_dirname()
        self.output.dirpath = os.path.join(
            self.output.root,
            "llm",
            self.model.family,
            self.model.name,
            *self.quant.generate_dirnames(default_dtype=self.model.dtype)[:-1],
            self.quant.generate_calib_dirname(),
            self.output.dirname,
        )
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)

    @classmethod
    def get_parser(cls) -> ConfigParser:
        """Get a parser for evaluating a large language model.

        Returns:
            `ConfigParser`: A parser for evaluating a large language model.
        """
        parser = ConfigParser("Evaluate a large language model")
        parser.add_config(cls)
        return parser
