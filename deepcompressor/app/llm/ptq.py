# -*- coding: utf-8 -*-
"""Evaluate a large language model."""

import gc
import json
import os
import pprint
import traceback
import time
import shutil

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer

from deepcompressor.utils import tools

from .config import LlmCacheConfig, LlmPtqRunConfig, LlmQuantCacheConfig, LlmQuantConfig
from .nn import LlmModelStruct
from .quant import quantize_llm_activations, quantize_llm_weights, reorder_llm, rotate_llm, smooth_llm
from .quant.weight import quantize_llm_layer_weights
from deepcompressor.data.cache import IOTensorsCache, TensorCache
from deepcompressor.data.utils.reshape import LinearReshapeFn
from .quant.quantizer.config import LlmExtraWeightQuantizerConfig

__all__ = ["ptq"]


def _evaluate_metric(model, tokenizer, eval_cfg: "LlmEvalConfig", model_name: str, output_dirpath: str, metric: str) -> float:
    """Run the GPTQ-style evaluator and return either PPL or NLL on the first task.

    - ppl: returns word_perplexity as reported.
    - nll: returns ln(word_perplexity).
    """
    from .eval.config import LlmEvalConfig as _LlmEvalConfig

    assert isinstance(eval_cfg, _LlmEvalConfig)
    # Prefer eos from model.config, then tokenizer, then GenerationConfig as fallback
    try:
        eos_token_ids = None
        if hasattr(model, "config") and getattr(model.config, "eos_token_id", None) is not None:
            eos_token_ids = model.config.eos_token_id
        elif getattr(tokenizer, "eos_token_id", None) is not None:
            eos_token_ids = tokenizer.eos_token_id
        else:
            source = getattr(model, "name_or_path", None) or getattr(tokenizer, "name_or_path", None)
            if source:
                eos_token_ids = GenerationConfig.from_pretrained(source).eos_token_id
        if eos_token_ids is None:
            eos_token_ids = []
        elif not isinstance(eos_token_ids, (list, tuple)):
            eos_token_ids = [eos_token_ids]
    except Exception:
        eos_token_ids = []
    results = eval_cfg.evaluate(
        model,
        tokenizer,
        model_name=model_name,
        eos_token_ids=eos_token_ids,
        output_dirpath=output_dirpath,
    )
    ev = "gptq" if "gptq" in results else next(iter(results))
    maxlen_bucket = next(iter(results[ev].keys()))
    task_name = next(iter(results[ev][maxlen_bucket]["results"].keys()))
    ppl = float(results[ev][maxlen_bucket]["results"][task_name]["word_perplexity"])
    if metric.lower() == "nll":
        import math as _math
        return float(_math.log(ppl))
    return ppl


def _first_supported_task(tasks: list[str]) -> str:
    for t in tasks:
        if t.startswith("wikitext"):
            return t
        if t.startswith("pile"):
            return t
    return "wikitext-2-raw-v1"


def _compute_token_nlls(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: str,
    seq_length: int,
    max_num_samples: int = -1,
    batch_size: int = 1,
) -> torch.Tensor:
    """Compute per-token NLLs over contiguous windows.

    Notes:
    - Tokenization is cached to disk to avoid repeated work across calls.
    - Tokens stay on CPU; windows are transferred to the model device.
    - Processes multiple windows per forward via `batch_size` for throughput.
    - KV cache is disabled during forward for efficiency.
    """
    from datasets import load_dataset

    logger = tools.logging.getLogger(f"{__name__}.TokenNLLs")
    model.eval()

    # Cache tokenized test set per (task, tokenizer) on disk (prefer /dev/shm when available)
    tok_id = getattr(tokenizer, "name_or_path", None) or getattr(tokenizer, "init_kwargs", {}).get("name_or_path", "tok")
    tok_id = str(tok_id).replace(os.sep, "_")
    cache_root = os.environ.get("DEEPCOMPRESSOR_TOKEN_CACHE_DIR", "/dev/shm/sonpt/token_cache")
    os.makedirs(cache_root, exist_ok=True)
    cache_file = os.path.join(cache_root, f"{task.replace('/', '_')}_{tok_id}.pt")

    encoded: torch.Tensor | None = None
    if os.path.exists(cache_file):
        try:
            encoded = torch.load(cache_file, map_location="cpu")
        except Exception:
            encoded = None

    if encoded is None:
        if task.startswith("wikitext"):
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            text = "\n\n".join(ds["text"])
        elif task.startswith("pile"):
            ds = load_dataset("pile", task, split="test")
            text = "\n\n".join(ds["text"])
        else:
            raise ValueError(f"Unsupported task for token NLLs: {task}")
        encoded = tokenizer(text, return_tensors="pt").input_ids.cpu()
        try:
            torch.save(encoded, cache_file)
        except Exception:
            pass

    total_tokens = int(encoded.numel())
    num_samples = total_tokens // seq_length
    # Optional env override for max samples to accelerate scans
    try:
        env_max = int(os.environ.get("DEEPCOMPRESSOR_TOKEN_NLL_MAX_SAMPLES", "0"))
        if env_max > 0:
            max_num_samples = env_max if max_num_samples <= 0 else min(max_num_samples, env_max)
    except Exception:
        pass
    if max_num_samples > 0:
        num_samples = min(num_samples, max_num_samples)

    if num_samples <= 0:
        return torch.empty(0)

    logger.debug(
        "Token NLLs: task=%s, tokens=%d, seq_len=%d, samples=%d",
        task,
        total_tokens,
        seq_length,
        num_samples,
    )

    # Prepare views and buffers
    window_len = seq_length - 1
    out = torch.empty(num_samples * window_len, dtype=torch.float32)
    encoded = encoded[:, : num_samples * seq_length]
    windows = encoded.view(num_samples, seq_length)
    if windows.device.type == "cpu" and torch.cuda.is_available():
        windows = windows.pin_memory()

    # Ensure sensible batch size; allow env override for convenience
    try:
        env_bs = int(os.environ.get("DEEPCOMPRESSOR_TOKEN_NLL_BATCH_SIZE", "0"))
        if env_bs > 0:
            batch_size = env_bs
    except Exception:
        pass
    batch_size = max(1, int(batch_size))
    num_batches = (num_samples + batch_size - 1) // batch_size

    t0 = time.perf_counter()

    with torch.inference_mode():
        for b in tqdm(range(num_batches), total=num_batches, desc="token NLL", dynamic_ncols=True, leave=False):
            i = b * batch_size
            cur_bs = min(batch_size, num_samples - i)
            batch = windows[i : i + cur_bs].to(model.device, non_blocking=True)
            try:
                logits = model(batch, use_cache=False).logits
            except TypeError:
                logits = model(batch).logits
            logits = logits[:, :-1, :].contiguous()
            labels = batch[:, 1:].contiguous()
            # per-token negative log-likelihood
            pt = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1), reduction="none")
            start = i * window_len
            out[start : start + cur_bs * window_len] = pt.detach().cpu()
    t1 = time.perf_counter()
    # Debug throughput
    if t1 > t0:
        tok_count = num_samples * window_len
        logger.debug(
            "Token NLLs done: samples=%d, elapsed=%.2fs, tok/s=%.1f",
            num_samples,
            t1 - t0,
            tok_count / (t1 - t0),
        )

    return out


def _run_single_layer_delta_scan(model, tokenizer, cfg: "LlmPtqRunConfig", logging_level: int) -> None:  # noqa: C901
    """Compute ΔLoss per submodule field by quantizing only that field at a time.

    Records JSON to output dir: {"baseline": ppl, "fields": {field: {"ppl": v, "delta": v-baseline}}}
    """
    logger = tools.logging.getLogger(f"{__name__}.DeltaLoss")
    logger.info("=== Single-Layer ΔLoss Scan ===")
    out_dir = cfg.output.get_running_job_path("delta")
    os.makedirs(out_dir, exist_ok=True)
    # Metric selection: preserve explicit request; support "auto" -> paired NLL
    chosen_metric = (cfg.delta_metric or "").lower().strip()
    if chosen_metric in ("", "auto"):
        metric_name = "nll_paired"
    elif chosen_metric in ("nll", "nll_paired"):
        metric_name = "nll_paired"
    elif chosen_metric == "ppl":
        metric_name = "ppl"
    else:
        tools.logging.getLogger(f"{__name__}.DeltaLoss").warning(
            "Unknown delta_metric '%s', defaulting to ppl", chosen_metric
        )
        metric_name = "ppl"

    # 1) Baseline (all high precision)
    logger.info("* Evaluating baseline (all high)")
    tools.logging.Formatter.indent_inc()
    # If paired NLL, compute baseline per-token NLLs once
    baseline_tokens = None
    if metric_name == "nll_paired":
        from .eval.config import get_max_seq_length as _get_max_seq_length
        lm_max = _get_max_seq_length(model, tokenizer)
        seq_len = lm_max if cfg.eval.max_seq_length in (0, -1) else min(lm_max, abs(cfg.eval.max_seq_length) or 2048)
        task = _first_supported_task(cfg.eval.tasks)
        baseline_tokens = _compute_token_nlls(
            model,
            tokenizer,
            task,
            seq_len,
            batch_size=max(1, getattr(cfg.eval, "batch_size", 1)),
        )
        baseline_val = float(baseline_tokens.mean().item()) if baseline_tokens.numel() > 0 else float("nan")
    else:
        baseline_val = _evaluate_metric(model, tokenizer, cfg.eval, cfg.model.name + "-baseline", out_dir, metric_name)
    tools.logging.Formatter.indent_dec()
    logger.info(f"  baseline {cfg.delta_metric}: {baseline_val:.6f}")

    # 2) For each layer and each selected linear submodule, quantize only that one and evaluate
    results = {"metric": metric_name, "baseline": baseline_val, "layers": []}
    # Helper: filter function for requested submodule roles
    field_aliases = set(cfg.delta_fields)
    def _want(field_name: str, module_name: str) -> bool:
        if not field_aliases:
            return True
        if field_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            return field_name in field_aliases
        if field_name == "down_proj":
            return ("down" in field_aliases) or ("down_proj" in field_aliases)
        if field_name == "up_proj":
            # Distinguish up vs gate by module name suffix when possible
            is_gate = module_name.endswith((".gate_proj", ".wi_0", ".w1"))
            is_up = module_name.endswith((".up_proj", ".wi_1", ".w3")) or not is_gate
            if ("gate" in field_aliases) or ("gate_proj" in field_aliases):
                return is_gate
            if ("up" in field_aliases) or ("up_proj" in field_aliases):
                return is_up
            return False
        if field_name == "moe_gate":
            return ("gate" in field_aliases) or ("moe_gate" in field_aliases)
        # proj_in / proj_out
        if field_name == "proj_in":
            return ("proj_in" in field_aliases)
        if field_name == "proj_out":
            return ("proj_out" in field_aliases)
        return False

    # Define which modules need inputs captured for calibration reuse
    import torch.nn as _nn
    def _needs_inputs(name: str, module: _nn.Module) -> bool:
        cls = module.__class__.__name__
        if isinstance(module, _nn.Linear):
            return True
        if cls in ("RotaryEmbedding", "MixtralSparseMoeBlock", "T5DenseActDense", "T5DenseGatedActDense"):
            return True
        if cls.endswith(("DecoderLayer", "Attention", "MLP")):
            return True
        return False

    # Iterate layers to collect per-layer module targets
    # Enumerate modules without running calibration to collect targets
    model_struct = LlmModelStruct.construct(model)
    layer_targets: list[tuple[int, str, str, str]] = []  # (layer_idx, module_key, module_name, field_name)
    for layer_idx, layer_struct in enumerate(model_struct.backbone_struct.layer_structs):
        for module_key, module_name, module, parent, field_name in layer_struct.named_key_modules():
            if getattr(module, "weight", None) is None:
                continue
            if not _want(field_name, module_name):
                continue
            layer_targets.append((layer_idx, module_key, module_name, field_name))

    # Precompute and cache per-layer activation inputs once (optional)
    act_cache_mode = (getattr(cfg, "delta_act_cache", "auto") or "auto").lower().strip()
    act_cache_dir_cfg = (getattr(cfg, "delta_act_cache_dir", "") or "").strip()
    if act_cache_dir_cfg == "" or act_cache_dir_cfg.lower() == "default":
        act_cache_dir = os.path.join(out_dir, "act_cache")
    elif act_cache_dir_cfg.lower() == "shm" or act_cache_dir_cfg.startswith("/dev/shm"):
        base = os.path.basename(out_dir.rstrip(os.sep)) or "delta"
        root = "/dev/shm" if act_cache_dir_cfg.lower() == "shm" else act_cache_dir_cfg
        act_cache_dir = os.path.join(root, "deepcompressor", "act_cache", base)
    else:
        act_cache_dir = act_cache_dir_cfg
    if act_cache_mode != "skip":
        os.makedirs(act_cache_dir, exist_ok=True)
        target_layer_idxs = sorted({idx for idx, _, _, _ in layer_targets})
        logger.info("* Collecting activation caches (one pass)")
        tools.logging.Formatter.indent_inc()
        try:
            calib_loader = cfg.quant.calib.build_loader(tokenizer)
            with tools.logging.redirect_tqdm():
                for idx, (_, (layer_struct, layer_cache, layer_kwargs)) in enumerate(
                    calib_loader.iter_layer_activations(
                        model_struct,
                        needs_inputs_fn=_needs_inputs,
                    )
                ):
                    if idx not in target_layer_idxs:
                        continue
                    save_path = os.path.join(act_cache_dir, f"L{idx}.pt")
                    if act_cache_mode == "auto" and os.path.exists(save_path):
                        continue
                    # Filter to only needed module names for this layer
                    wanted_names = set()
                    for _, module_name, module, parent, field_name in layer_struct.named_key_modules():
                        if getattr(module, "weight", None) is None:
                            continue
                        if _want(field_name, module_name):
                            wanted_names.add(module_name)
                            # Also keep attention module inputs for q/k eval
                            if field_name in ("q_proj", "k_proj") and hasattr(parent, "name"):
                                wanted_names.add(parent.name)
                    pack = {"kwargs": layer_kwargs, "inputs": {}}
                    for name, io in layer_cache.items():
                        if name in wanted_names and getattr(io, "inputs", None) is not None:
                            # Store standardized inputs to reduce size
                            try:
                                data = [t.cpu() for t in io.inputs.get_standardized_data(reshape=False)]
                            except Exception:
                                # Fallback to raw tensors
                                data = [t.cpu() for t in io.inputs.front().data] if io.inputs is not None else []
                            pack["inputs"][name] = data
                    torch.save(pack, save_path)
        finally:
            tools.logging.Formatter.indent_dec()

    # Free the baseline model/tokenizer to release GPU memory before scanning layers
    try:
        model.to("cpu")
    except Exception:
        pass
    del model_struct
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Free the baseline model/tokenizer to release GPU memory before scanning layers
    try:
        model.to("cpu")
    except Exception:
        pass
    del model_struct
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Evaluate each target by quantizing exactly that module
    for layer_idx, module_key, module_name, field_name in layer_targets:
        logger.info(f"* Layer {layer_idx}: {module_name}")
        tools.logging.Formatter.indent_inc()
        # Rebuild fresh model/tokenizer
        m, t = cfg.model.build()
        m_struct = LlmModelStruct.construct(m)
        # Prepare qcfg: disable base wgts, enable only this module via per-instance filter and (optionally) extra_wgts
        import copy as _copy
        qcfg_local = _copy.deepcopy(cfg.quant)
        qcfg_local.rotation = None
        qcfg_local.reorder = None
        qcfg_local.smooth = None
        # turn off base weights
        qcfg_local.wgts.dtype = None
        # limit to this exact module name to ensure only this submodule quantizes
        qcfg_local.only_module_names = (module_name,)
        # Prepare extra wgts template (we will set includes after identifying the exact eff_key per layer)
        base = cfg.quant.wgts
        qcfg_local.extra_wgts = LlmExtraWeightQuantizerConfig(
            includes=[],
            dtype=base.dtype,
            zero_point=base.zero_point,
            group_shapes=base.group_shapes,
            scale_dtypes=base.scale_dtypes,
            intermediate_dtypes=getattr(base, "intermediate_dtypes", ()),
            intermediate_levels=getattr(base, "intermediate_levels", ()),
            needs_dequant_saturation=getattr(base, "needs_dequant_saturation", False),
            skips=[],
            kernel_gptq=base.kernel_gptq,
            calib_range=base.calib_range,
        )
        # Calibrate and quantize only the target layer using cached activations
        quant_done = False
        cache_path = os.path.join(act_cache_dir, f"L{layer_idx}.pt")
        lyr, lyr_cache, lyr_kwargs = None, None, None
        if act_cache_mode != "skip" and os.path.exists(cache_path):
            pack = torch.load(cache_path, map_location="cpu")
            lyr_kwargs = pack.get("kwargs", {})
            inputs_pack = pack.get("inputs", {})
            lyr_cache = {}
            for name, data in inputs_pack.items():
                lyr_cache[name] = IOTensorsCache(
                    inputs=TensorCache(data=[d for d in data], channels_dim=-1, reshape=LinearReshapeFn())
                )
            lyr = m_struct.backbone_struct.layer_structs[layer_idx]
        else:
            # Fallback: collect activations only up to this layer in this model
            import torch.nn as _nn
            def _needs_inputs(name: str, module: _nn.Module) -> bool:
                cls = module.__class__.__name__
                if isinstance(module, _nn.Linear):
                    return True
                if cls in ("RotaryEmbedding", "MixtralSparseMoeBlock", "T5DenseActDense", "T5DenseGatedActDense"):
                    return True
                if cls.endswith(("DecoderLayer", "Attention", "MLP")):
                    return True
                return False
            for idx, (_, (lyr, tmp_cache, lyr_kwargs)) in enumerate(
                qcfg_local.calib.build_loader(t).iter_layer_activations(
                    m_struct,
                    needs_inputs_fn=_needs_inputs,
                )
            ):
                if idx == layer_idx:
                    lyr_cache = tmp_cache
                    break
        if lyr_cache is None or lyr is None:
            logger.warning("  could not load/collect layer cache for L%d %s", layer_idx, module_name)
            tools.logging.Formatter.indent_dec()
            continue
        # Identify the effective include key for this module instance within this layer
        eff_key = module_key
        if field_name.endswith("_proj") or field_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            # Try to match against attention names
            for attn in getattr(lyr, "attn_structs", []):
                if field_name == "q_proj" and attn.q_proj_name == module_name:
                    eff_key = attn.q_key
                    break
                if field_name == "k_proj" and attn.k_proj_name == module_name:
                    eff_key = attn.k_key
                    break
                if field_name == "v_proj" and attn.v_proj_name == module_name:
                    eff_key = attn.v_key
                    break
                if field_name == "o_proj" and attn.o_proj_name == module_name:
                    eff_key = attn.out_proj_key
                    break
            # Match FFN projections
            ffn = getattr(lyr, "ffn_struct", None)
            if ffn is not None and eff_key == module_key:
                if field_name == "up_proj" and module_name in ffn.up_proj_names:
                    eff_key = ffn.up_proj_key
                elif field_name == "down_proj" and module_name in ffn.down_proj_names:
                    eff_key = ffn.down_proj_key
                elif field_name == "moe_gate" and getattr(ffn, "moe_gate_name", None) == module_name:
                    eff_key = ffn.moe_gate_key
        # Set includes to the identified eff_key
        qcfg_local.extra_wgts.includes = [eff_key]
        quantize_llm_layer_weights(
            layer=lyr,
            config=qcfg_local,
            quantizer_state_dict={},
            layer_cache=lyr_cache,
            layer_kwargs=lyr_kwargs,
            return_with_scale_state_dict=False,
        )
        quant_done = True
        if not quant_done:
            logger.warning("  could not quantize layer %d for %s", layer_idx, module_name)
            tools.logging.Formatter.indent_dec()
            continue
        if metric_name == "nll_paired":
            # Paired per-token NLL difference
            from .eval.config import get_max_seq_length as _get_max_seq_length
            lm_max = _get_max_seq_length(m, t)
            seq_len = lm_max if cfg.eval.max_seq_length in (0, -1) else min(lm_max, abs(cfg.eval.max_seq_length) or 2048)
            task = _first_supported_task(cfg.eval.tasks)
            model_tokens = _compute_token_nlls(
                m,
                t,
                task,
                seq_len,
                batch_size=max(1, getattr(cfg.eval, "batch_size", 1)),
            )
            # align lengths just in case
            L = min(baseline_tokens.numel(), model_tokens.numel()) if baseline_tokens is not None else 0
            paired_delta = (model_tokens[:L] - baseline_tokens[:L]).mean().item() if L > 0 else float("nan")
            val = float(model_tokens[:L].mean().item()) if L > 0 else float("nan")
            delta = float(paired_delta)
        else:
            val = _evaluate_metric(m, t, cfg.eval, cfg.model.name + f"-L{layer_idx}-{module_name}", out_dir, metric_name)
            delta = val - baseline_val
        # Accumulate per-layer results
        while len(results["layers"]) <= layer_idx:
            results["layers"].append({"modules": []})
        results["layers"][layer_idx]["modules"].append({
            "module": module_name,
            "key": module_key,
            metric_name: val,
            "delta": delta,
        })
        logger.info(f"  {metric_name}: {val:.6f} | Δ: {delta:.6f}")
        tools.logging.Formatter.indent_dec()
        # Cleanup rebuilt model to free GPU memory before next layer
        try:
            m.to("cpu")
        except Exception:
            pass
        del m_struct, m, t
        gc.collect()
        torch.cuda.empty_cache()

    with open(os.path.join(out_dir, "delta_loss.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved ΔLoss results to {out_dir}")
    # Optionally delete activation cache directory
    try:
        if (getattr(cfg, "delta_act_cache", "auto") or "auto").lower().strip() != "skip" and getattr(cfg, "delta_act_cache_cleanup", False):
            if os.path.isdir(act_cache_dir):
                shutil.rmtree(act_cache_dir, ignore_errors=True)
                logger.info(f"Deleted activation cache at {act_cache_dir}")
    except Exception:
        logger.warning("Failed to delete activation cache directory", exc_info=True)
    gc.collect()
    torch.cuda.empty_cache()


def ptq(  # noqa: C901
    model: PreTrainedModel | LlmModelStruct,
    /,
    tokenizer: PreTrainedTokenizer,
    config: LlmQuantConfig,
    cache: LlmCacheConfig | None = None,
    load_dirpath: str = "",
    save_dirpath: str = "",
    copy_on_save: bool = False,
    save_model: bool = False,
) -> PreTrainedModel:
    """Post-training quantization of a large language model.

    Args:
        model (`PreTrainedModel` or `LlmStruct`):
            The large language model.
        tokenizer (`PreTrainedTokenizer`):
            The large language model tokenizer.
        config (`LlmQuantConfig`):
            The large language model post-training quantization configuration.
        cache (`LlmCacheConfig`, *optional*, defaults to `None`):
            The large language model quantization cache path configuration.
        load_dirpath (`str`, *optional*, defaults to `""`):
            The directory path to load the quantization checkpoint.
        save_dirpath (`str`, *optional*, defaults to `""`):
            The directory path to save the quantization checkpoint.
        copy_on_save (`bool`, *optional*, defaults to `False`):
            Whether to copy the cache to the save directory.
        save_model (`bool`, *optional*, defaults to `False`):
            Whether to save the quantized model checkpoint.

    Returns:
        `PreTrainedModel`:
            The quantized model.
    """
    logger = tools.logging.getLogger(__name__)
    if not isinstance(model, LlmModelStruct):
        model = LlmModelStruct.construct(model)
    assert isinstance(model, LlmModelStruct)

    quant_wgts = config.enabled_wgts
    quant_ipts = config.enabled_ipts
    quant_opts = config.enabled_opts
    quant_acts = quant_ipts or quant_opts
    quant = quant_wgts or quant_acts
    needs_rotation = quant and config.enabled_rotation
    needs_reorder = quant and config.enabled_reorder
    needs_smooth = quant and config.enabled_smooth

    load_model_path, load_path, save_path = "", None, None
    if load_dirpath:
        load_path = LlmQuantCacheConfig(
            rotation=os.path.join(load_dirpath, "rotation.pt"),
            reorder=os.path.join(load_dirpath, "reorder.pt"),
            smooth=os.path.join(load_dirpath, "smooth.pt"),
            wgts=os.path.join(load_dirpath, "wgts.pt"),
            acts=os.path.join(load_dirpath, "acts.pt"),
        )
        load_model_path = os.path.join(load_dirpath, "model.pt")
        if os.path.exists(load_model_path):
            logger.info(f"* Found the model from {load_model_path}")
            load_model = True
            save_dirpath = ""  # do not save the model if loading
            if needs_reorder and not config.reorder.dynamic:
                needs_reorder = False
                logger.info("* Safe to skip reordering the model")
            if needs_smooth:
                needs_smooth = False
                logger.info("* Safe to skip smoothing the model")
        else:
            logger.warning(f"Model checkpoint {load_model_path} does not exist")
            load_model, load_model_path = False, ""
    else:
        load_model = False
    if save_dirpath:
        os.makedirs(save_dirpath, exist_ok=True)
        save_path = LlmQuantCacheConfig(
            rotation=os.path.join(save_dirpath, "rotation.pt"),
            reorder=os.path.join(save_dirpath, "reorder.pt"),
            smooth=os.path.join(save_dirpath, "smooth.pt"),
            wgts=os.path.join(save_dirpath, "wgts.pt"),
            acts=os.path.join(save_dirpath, "acts.pt"),
        )
    else:
        save_model = False

    # region rotate model
    if needs_rotation:
        logger.info("* Rotating model")
        tools.logging.Formatter.indent_inc()
        load_from = ""
        if load_path and os.path.exists(load_path.rotation):
            load_from = load_path.rotation
        elif cache and cache.path.rotation and os.path.exists(cache.path.rotation):
            load_from = cache.path.rotation
        elif os.path.exists(config.rotation.path):
            load_from = config.rotation.path
        if load_from:
            logger.info(f"- Loading rotation from {load_from}")
            rotation = torch.load(load_from).to(dtype=torch.float64)
            rotate_llm(model, config.rotation, rotation=rotation)
        else:
            logger.info("- Generating rotation")
            rotation = rotate_llm(model, config.rotation)
            if cache and cache.path.rotation:
                logger.info(f"- Saving rotation to {cache.path.rotation}")
                os.makedirs(cache.dirpath.rotation, exist_ok=True)
                torch.save(rotation, cache.path.rotation)
                load_from = cache.path.rotation
        if save_path:
            if not copy_on_save and load_from:
                logger.info(f"- Linking rotation to {save_path.rotation}")
                os.symlink(os.path.relpath(load_from, save_dirpath), save_path.rotation)
            else:
                logger.info(f"- Saving rotation to {save_path.rotation}")
                torch.save(rotation, save_path.rotation)
        del rotation
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()
    logger.info(f"* Development dtype is {config.develop_dtype}")
    # endregion
    # region reorder channels
    if needs_reorder:
        logger.info("* Reordering channels")
        tools.logging.Formatter.indent_inc()
        load_from = ""
        if load_path and os.path.exists(load_path.reorder):
            load_from = load_path.reorder
        elif cache and cache.path.reorder and os.path.exists(cache.path.reorder):
            load_from = cache.path.reorder
        if load_from:
            logger.info(f"- Loading reorder indices from {load_from}")
            reorder_cache = torch.load(load_from)
            reorder_llm(model, config, tokenizer, reorder_cache=reorder_cache)
        else:
            logger.info("- Generating reorder indices")
            reorder_cache = reorder_llm(model, config, tokenizer)
            if cache and cache.path.reorder:
                logger.info(f"- Saving reorder indices to {cache.path.reorder}")
                os.makedirs(cache.dirpath.reorder, exist_ok=True)
                torch.save(reorder_cache, cache.path.reorder)
                load_from = cache.path.reorder
        if save_path:
            if not copy_on_save and load_from:
                logger.info(f"- Linking reorder indices to {save_path.reorder}")
                os.symlink(os.path.relpath(load_from, save_dirpath), save_path.reorder)
            else:
                logger.info(f"- Saving reorder indices to {save_path.reorder}")
                torch.save(reorder_cache, save_path.reorder)
        del reorder_cache
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()
    # endregion
    # region smooth quantization
    if needs_smooth:
        logger.info("* Smoothing model for quantization")
        tools.logging.Formatter.indent_inc()
        load_from = ""
        if load_path and os.path.exists(load_path.smooth):
            load_from = load_path.smooth
        elif cache and cache.path.smooth and os.path.exists(cache.path.smooth):
            load_from = cache.path.smooth
        if load_from:
            logger.info(f"- Loading smooth scales from {load_from}")
            smooth_cache = torch.load(load_from)
            smooth_llm(model, config, smooth_cache=smooth_cache)
        else:
            logger.info("- Generating smooth scales")
            smooth_cache = smooth_llm(model, config, tokenizer=tokenizer)
            if cache and cache.path.smooth:
                logger.info(f"- Saving smooth scales to {cache.path.smooth}")
                os.makedirs(cache.dirpath.smooth, exist_ok=True)
                torch.save(smooth_cache, cache.path.smooth)
                load_from = cache.path.smooth
        if save_path:
            if not copy_on_save and load_from:
                logger.info(f"- Linking smooth scales to {save_path.smooth}")
                os.symlink(os.path.relpath(load_from, save_dirpath), save_path.smooth)
            else:
                logger.info(f"- Saving smooth scales to {save_path.smooth}")
                torch.save(smooth_cache, save_path.smooth)
        del smooth_cache
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()
    # endregion
    # region collect original state dict
    if config.needs_acts_quantizer_cache:
        if load_path and os.path.exists(load_path.acts):
            orig_state_dict = None
        elif cache and cache.path.acts and os.path.exists(cache.path.acts):
            orig_state_dict = None
        else:
            orig_state_dict: dict[str, torch.Tensor] = {
                name: param.detach().clone() for name, param in model.module.named_parameters() if param.ndim > 1
            }
    else:
        orig_state_dict = None
    # endregion
    if load_model:
        logger.info(f"* Loading model checkpoint from {load_model_path}")
        model.module.load_state_dict(torch.load(load_model_path))
        gc.collect()
        torch.cuda.empty_cache()
    elif quant_wgts:
        logger.info("* Quantizing weights")
        tools.logging.Formatter.indent_inc()
        load_from = ""
        if load_path and os.path.exists(load_path.wgts):
            load_from = load_path.wgts
        elif cache and cache.path.wgts and os.path.exists(cache.path.wgts):
            load_from = cache.path.wgts
        if load_from:
            logger.info(f"- Loading weight quantizer settings from {load_from}")
            quantizer_state_dict = torch.load(load_from)
            _, scale_state_dict = quantize_llm_weights(
                model,
                config,
                tokenizer=tokenizer,
                quantizer_state_dict=quantizer_state_dict,
                return_with_scale_state_dict=save_model,
            )
        else:
            logger.info("- Generating weight quantizer settings")
            quantizer_state_dict, scale_state_dict = quantize_llm_weights(
                model, config, tokenizer=tokenizer, return_with_scale_state_dict=save_model
            )
            if cache and cache.dirpath.wgts:
                logger.info(f"- Saving weight quantizer settings to {cache.path.wgts}")
                os.makedirs(cache.dirpath.wgts, exist_ok=True)
                torch.save(quantizer_state_dict, cache.path.wgts)
                load_from = cache.path.wgts
        if save_path:
            if not copy_on_save and load_from:
                logger.info(f"- Linking weight quantizer settings to {save_path.wgts}")
                os.symlink(os.path.relpath(load_from, save_dirpath), save_path.wgts)
            else:
                logger.info(f"- Saving weight quantizer settings to {save_path.wgts}")
                torch.save(quantizer_state_dict, save_path.wgts)
        if save_model:
            logger.info(f"- Saving model checkpoint to {save_dirpath}")
            torch.save(scale_state_dict, os.path.join(save_dirpath, "scale.pt"))
            torch.save(model.module.state_dict(), os.path.join(save_dirpath, "model.pt"))
        del quantizer_state_dict, scale_state_dict
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()
    if quant_acts:
        logger.info("  * Quantizing activations")
        tools.logging.Formatter.indent_inc()
        if config.needs_acts_quantizer_cache:
            load_from = ""
            if load_path and os.path.exists(load_path.acts):
                load_from = load_path.acts
            elif cache and cache.path.acts and os.path.exists(cache.path.acts):
                load_from = cache.path.acts
            if load_from:
                logger.info(f"- Loading activation quantizer settings from {load_from}")
                quantizer_state_dict = torch.load(load_from)
                quantize_llm_activations(
                    model,
                    config,
                    tokenizer=tokenizer,
                    quantizer_state_dict=quantizer_state_dict,
                    orig_state_dict=orig_state_dict,
                )
            else:
                logger.info("- Generating activation quantizer settings")
                quantizer_state_dict = quantize_llm_activations(
                    model, config, tokenizer=tokenizer, orig_state_dict=orig_state_dict
                )
                if cache and cache.dirpath.acts:
                    logger.info(f"- Saving activation quantizer settings to {cache.path.acts}")
                    os.makedirs(cache.dirpath.acts, exist_ok=True)
                    torch.save(quantizer_state_dict, cache.path.acts)
                    load_from = cache.path.acts
            if save_dirpath:
                if not copy_on_save and load_from:
                    logger.info(f"- Linking activation quantizer settings to {save_path.acts}")
                    os.symlink(os.path.relpath(load_from, save_dirpath), save_path.acts)
                else:
                    logger.info(f"- Saving activation quantizer settings to {save_path.acts}")
                    torch.save(quantizer_state_dict, save_path.acts)
            del quantizer_state_dict
        else:
            logger.info("- No need to generate/load activation quantizer settings")
            quantize_llm_activations(model, config, tokenizer=tokenizer, orig_state_dict=orig_state_dict)
        tools.logging.Formatter.indent_dec()
        del orig_state_dict
        gc.collect()
        torch.cuda.empty_cache()
    return model.module


def main(config: LlmPtqRunConfig, logging_level: int = tools.logging.DEBUG) -> None:  # noqa: C901
    """Post-training quantization and evaluation of a large language model.

    Args:
        config (`LlmPtqConfig`):
            The large language model post-training quantization configuration.
        logging_level (`int`, *optional*, defaults to `logging.DEBUG`):
            The logging level.
    """
    config.output.lock()
    config.dump(path=config.output.get_running_job_path("config.yaml"))
    tools.logging.setup(path=config.output.get_running_job_path("run.log"), level=logging_level)
    logger = tools.logging.getLogger(__name__)
    # region log configurations
    logger.info("=== Configurations ===")
    tools.logging.info(config.formatted_str(), logger=logger)
    logger.info("=== Dumped Configurations ===")
    tools.logging.info(pprint.pformat(config.dump(), indent=2, width=120), logger=logger)
    logger.info("=== Output Directory ===")
    logger.info(config.output.job_dirpath)
    # endregion
    logger.info("=== Start Evaluating ===")
    logger.info(f"* Building model {config.model.name} from {config.model.path}")
    tools.logging.Formatter.indent_inc()
    model, tokenizer = config.model.build()
    tools.logging.Formatter.indent_dec()
    # Delta-loss mode: compute baseline and per-field deltas, then exit
    if getattr(config, "delta_single_layer", False):
        _run_single_layer_delta_scan(model, tokenizer, config, logging_level)
        config.output.unlock()
        return
    save_dirpath = os.path.join(config.output.running_job_dirpath, "cache")
    if config.save_model:
        if config.save_model.lower() in ("false", "none", "null", "nil"):
            save_model = False
        elif config.save_model.lower() in ("true", "default"):
            save_dirpath, save_model = os.path.join(config.output.running_job_dirpath, "model"), True
        else:
            save_dirpath, save_model = config.save_model, True
    else:
        save_model = False
    model = ptq(
        model,
        tokenizer=tokenizer,
        config=config.quant,
        cache=config.cache,
        load_dirpath=config.load_from,
        save_dirpath=save_dirpath,
        copy_on_save=config.copy_on_save,
        save_model=save_model,
    )
    # region evaluate model
    if not config.skip_eval:
        logger.info("* Evaluating model")
        eos_token_ids = GenerationConfig.from_pretrained(config.model.path).eos_token_id
        if not isinstance(eos_token_ids, list):
            eos_token_ids = [eos_token_ids]
        tools.logging.Formatter.indent_inc()
        results = config.eval.evaluate(
            model,
            tokenizer,
            model_name=config.model.name,
            eos_token_ids=eos_token_ids,
            output_dirpath=config.output.get_running_job_path("eval"),
        )
        tools.logging.Formatter.indent_dec()
        logger.info(f"* Saving results to {config.output.job_dirpath}")
        # dump results
        with open(os.path.join(config.output.get_running_job_path("results.json")), "w") as f:
            json.dump(results, f, indent=2)
        # endregion
    config.output.unlock()


if __name__ == "__main__":
    config, _, unused_cfgs, unused_args, unknown_args = LlmPtqRunConfig.get_parser().parse_known_args()
    if len(unused_cfgs) > 0:
        tools.logging.warning(f"Unused configurations: {unused_cfgs}")
    if unused_args is not None:
        tools.logging.warning(f"Unused arguments: {unused_args}")
    assert len(unknown_args) == 0, f"Unknown arguments: {unknown_args}"
    try:
        main(config, logging_level=tools.logging.DEBUG)
    except Exception as e:
        tools.logging.Formatter.indent_reset()
        tools.logging.error("=== Error ===")
        tools.logging.error(traceback.format_exc())
        tools.logging.shutdown()
        traceback.print_exc()
        config.output.unlock(error=True)
        raise e
