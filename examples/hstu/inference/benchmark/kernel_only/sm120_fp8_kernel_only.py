#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""SM120 FP8 HSTU kernel-only benchmark.

This file is the customer-reproducible SM120 benchmark used for kernel-only
FP8 measurements. It keeps the validated benchmark logic and only parameterizes
the FBGEMM HSTU import path through FBGEMM_HSTU_PATH.
Measures kernel-only and end-to-end throughput for BF16 and FP8 block-scale
(quant_mode=2) attention on Blackwell RTX.

Kernel-only timing bypasses the Python-level quantization overhead to measure
the raw CUDA kernel performance (TFLOPS).

Usage:
    python sm120_fp8_kernel_only.py
    python sm120_fp8_kernel_only.py --mode kernel
    python sm120_fp8_kernel_only.py --mode e2e
    python sm120_fp8_kernel_only.py --mode all
    python sm120_fp8_kernel_only.py --seqlens 512 1024 2048 4096
    python sm120_fp8_kernel_only.py --mode kernel --mask-configs all --bias-configs all
    python sm120_fp8_kernel_only.py --mode kernel --mask-configs full causal --bias-configs none rab
    python sm120_fp8_kernel_only.py --mode kernel --columns fp8 paged
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_FBGEMM_HSTU_PATH = (
    REPO_ROOT / "third_party" / "FBGEMM" / "fbgemm_gpu" / "experimental" / "hstu"
)
FBGEMM_HSTU_PATH = Path(
    os.environ.get("FBGEMM_HSTU_PATH", str(DEFAULT_FBGEMM_HSTU_PATH))
).resolve()
FBGEMM_HSTU_BUILD_LIB = os.environ.get("FBGEMM_HSTU_BUILD_LIB")

import_paths = []
if FBGEMM_HSTU_BUILD_LIB:
    import_paths.append(Path(FBGEMM_HSTU_BUILD_LIB).resolve())
import_paths.extend([FBGEMM_HSTU_PATH, FBGEMM_HSTU_PATH / "test"])
sys.path[:0] = [str(path) for path in import_paths]

try:
    from hstu.cuda_hstu_attention import (
        get_bm_and_bn_block_size_fwd,
        quantize_for_block_scale,
        quantize_paged_kv_cache_for_block_scale,
        pack_descale_to_e8m0x4_int32,
    )
    from hstu_test import generate_input
    import hstu  # noqa: F401
except ImportError as e:
    print(f"ERROR: Failed to import hstu: {e}", file=sys.stderr)
    sys.exit(1)


WARMUP = 10
ITERS = 50
MASK_CONFIGS = ("full", "causal", "local", "context", "target", "arbitrary")
BIAS_CONFIGS = ("none", "rab", "drab")
COLUMNS = ("bf16", "fp8", "paged")
RAB_HEAD_MODES = ("per-head", "shared")


@dataclass(frozen=True)
class BenchCase:
    mask: str
    bias: str = "none"

    @property
    def has_rab(self) -> bool:
        return self.bias != "none"

    @property
    def has_drab(self) -> bool:
        return self.bias == "drab"

    @property
    def label(self) -> str:
        return self.mask if self.bias == "none" else f"{self.mask}+{self.bias}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_blackwell_rtx() -> None:
    assert torch.cuda.is_available(), "CUDA not available"
    major, minor = torch.cuda.get_device_capability()
    if major < 12:
        print(
            f"WARNING: Blackwell RTX kernels require SM12.x (RTX Pro 6000 Blackwell). "
            f"Detected SM{major}{minor}. FP8 block-scale benchmarks may fail.",
            file=sys.stderr,
        )


def _case_shape(case: BenchCase, seqlen: int) -> Tuple[int, int, Tuple[int, int], bool]:
    if case.mask == "full":
        return 0, 0, (-1, -1), False
    if case.mask == "causal":
        return 0, 0, (-1, 0), False
    if case.mask == "local":
        return 0, 0, (seqlen // 2, 16), False
    if case.mask == "context":
        return seqlen, 0, (-1, 0), False
    if case.mask == "target":
        return 0, seqlen, (-1, 0), False
    if case.mask == "arbitrary":
        return 0, 0, (-1, -1), True
    raise ValueError(f"unknown mask config: {case.mask}")


def build_cases(mask_configs: Sequence[str], bias_configs: Sequence[str]) -> List[BenchCase]:
    return [BenchCase(mask, bias) for mask in mask_configs for bias in bias_configs]


def fp8_block_n(headdim: int, has_rab: bool = False) -> int:
    """Blackwell RTX FP8 kBlockN used by the current forward dispatcher."""
    _, bn = get_bm_and_bn_block_size_fwd(object() if has_rab else None, headdim)
    return bn


def bf16_unsupported_reason(headdim: int, case: BenchCase) -> str:
    """Return why the BF16 baseline is unavailable for this Blackwell RTX benchmark case."""
    if headdim not in (32, 64, 128, 256):
        return "current Blackwell RTX BF16 benchmark supports headDim 32, 64, 128, or 256"
    return ""


def column_unsupported_reason(column: str, headdim: int, case: BenchCase) -> str:
    """Return why a benchmark output column is unavailable for a logical case."""
    if column == "bf16":
        return bf16_unsupported_reason(headdim, case)
    if column == "fp8":
        if headdim not in (32, 64, 128, 256):
            return "Blackwell RTX FP8 block-scale benchmark supports headDim 32, 64, 128, or 256"
        return ""
    if column == "paged":
        if headdim not in (32, 64, 128, 256):
            return "Blackwell RTX FP8 paged KV benchmark supports headDim 32, 64, 128, or 256"
        if fp8_block_n(headdim, case.has_rab) != 64:
            return "Blackwell RTX FP8 paged KV benchmark requires page_size=kBlockN=64"
        return ""
    raise ValueError(f"unknown benchmark column: {column}")


def has_supported_column(columns: Sequence[str], headdim: int, case: BenchCase) -> bool:
    return any(not column_unsupported_reason(column, headdim, case) for column in columns)


def print_unsupported_summary(
    headdims: Sequence[int],
    cases: Sequence[BenchCase],
    columns: Sequence[str],
) -> None:
    rows = []
    for column in columns:
        for headdim in headdims:
            grouped = {}
            for case in cases:
                reason = column_unsupported_reason(column, headdim, case)
                if reason:
                    grouped.setdefault(reason, []).append(case.label)
            for reason, labels in grouped.items():
                rows.append((column, headdim, reason, labels))

    print("Unsupported selected cases:")
    if not rows:
        print("  none")
        print()
        return
    for column, headdim, reason, labels in rows:
        print(
            f"  {column:>5} d={headdim}: {', '.join(labels)} "
            f"# {reason}"
        )
    print()


def fp8_round_bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float8_e4m3fn).to(torch.bfloat16)


def make_case_inputs(batch_size: int, seqlen: int, nheads: int,
                     headdim: int, case: BenchCase,
                     rab_heads: str = "per-head") -> dict:
    max_context_len, max_target_len, window_size, is_arbitrary = _case_shape(case, seqlen)
    heads_rab = 1 if case.has_rab and rab_heads == "shared" else None
    (
        _,
        _,
        num_contexts,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        num_targets,
        _,
        q,
        k,
        v,
        rab,
        attn_mask,
        func,
    ) = generate_input(
        batch_size=batch_size,
        heads=nheads,
        heads_rab=heads_rab,
        max_seq_len_q=seqlen,
        max_seq_len_k=seqlen,
        max_context_len=max_context_len,
        max_target_len=max_target_len,
        target_group_size=1,
        attn_dim=headdim,
        hidden_dim=headdim,
        window_size=window_size,
        dtype=torch.bfloat16,
        full_batch=True,
        has_drab=case.has_drab,
        is_delta_q=False,
        is_arbitrary=is_arbitrary,
    )

    max_seqlen_q = max_context_len + seqlen + max_target_len
    max_seqlen_k = max_context_len + seqlen + max_target_len
    if attn_mask is None:
        pairs = 0
        for b in range(batch_size):
            q_len = int((cu_seqlens_q[b + 1] - cu_seqlens_q[b]).item())
            k_len = int((cu_seqlens_k[b + 1] - cu_seqlens_k[b]).item())
            pairs += q_len * k_len
    else:
        pairs = int(attn_mask.bool().sum().item())

    return {
        "case": case,
        "rab_heads": rab_heads,
        "q": q.detach().contiguous(),
        "k": k.detach().contiguous(),
        "v": v.detach().contiguous(),
        "cu_q": cu_seqlens_q,
        "cu_k": cu_seqlens_k,
        "seqused_q": seqused_q,
        "seqused_k": seqused_k,
        "num_contexts": num_contexts,
        "num_targets": num_targets,
        "rab": rab.detach().to(torch.bfloat16).contiguous() if case.has_rab else None,
        "func": func,
        "window_size": window_size,
        "target_group_size": 1,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "pairs": pairs,
    }


def quantize_fp8_bs(q_bf16, k_bf16, v_bf16, cu_q, cu_k, headdim, has_rab=False):
    """Quantize BF16 Q/K/V to FP8 block-scale (quant_mode=2) format."""
    block_n = fp8_block_n(headdim, has_rab)

    # Round to FP8 range first (same as sweep_accuracy.py)
    q_in = fp8_round_bf16(q_bf16)
    k_in = fp8_round_bf16(k_bf16)
    v_in = fp8_round_bf16(v_bf16)

    # Q/K: block-scale along D (headdim axis)
    q_fp8, q_descale, cu_q_blk = quantize_for_block_scale(
        q_in, cu_q, fp8_type=torch.float8_e4m3fn,
        scale_mode="token_dchunk", d_chunk_size=128, round_to_e8m0=True
    )
    k_fp8, k_descale, cu_kv_blk = quantize_for_block_scale(
        k_in, cu_k, fp8_type=torch.float8_e4m3fn,
        scale_mode="token_dchunk", d_chunk_size=128, round_to_e8m0=True
    )

    # V: block-scale along N (sequence axis)
    v_fp8, v_descale, cu_v_blk = quantize_for_block_scale(
        v_in, cu_k, block_size=block_n, fp8_type=torch.float8_e4m3fn,
        scale_mode="seq_block", round_to_e8m0=True
    )
    # Pack descale factors to e8m0×4 int32 format for kernel.
    # sf_v is expanded from [H, total_blocks] → [H, total_tokens] via repeat_interleave
    # so TMA SFV can load kBlockN-element tiles indexed by nb_abs (same as SFB).
    sf_q = pack_descale_to_e8m0x4_int32(q_descale)
    sf_k = pack_descale_to_e8m0x4_int32(k_descale)
    sf_v = pack_descale_to_e8m0x4_int32(v_descale).repeat_interleave(block_n, dim=1)

    return (q_fp8, k_fp8, v_fp8,
            sf_q, sf_k, sf_v,
            q_descale, k_descale, v_descale,
            cu_q_blk, cu_kv_blk, cu_v_blk,
            cu_q, cu_k)


def make_paged_cache_from_varlen_kv(inputs: dict, page_size: int):
    k = inputs["k"]
    v = inputs["v"]
    cu_k = inputs["cu_k"]
    num_targets = inputs["num_targets"]
    batch_size = cu_k.numel() - 1
    target_lens = (
        num_targets
        if num_targets is not None
        else torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
    )
    lengths_k = cu_k[1:] - cu_k[:-1]
    cache_lens = (lengths_k - target_lens).to(torch.int32)
    if bool((cache_lens <= 0).any().item()):
        raise ValueError(f"paged benchmark requires positive cache length, got {cache_lens.cpu().tolist()}")

    pages_per_batch = (cache_lens + page_size - 1) // page_size
    page_offsets = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
    page_offsets[1:] = torch.cumsum(pages_per_batch, dim=0)
    total_pages = int(page_offsets[-1].item())
    page_ids = torch.randperm(total_pages, dtype=torch.int32, device="cuda")
    last_page_lens = ((cache_lens - 1) % page_size + 1).to(torch.int32)

    kv_cache = torch.zeros(
        (total_pages, 2, page_size, k.shape[1], k.shape[2]),
        dtype=k.dtype,
        device=k.device,
    )
    for b in range(batch_size):
        k0 = int(cu_k[b].item())
        cache_len = int(cache_lens[b].item())
        logical_page0 = int(page_offsets[b].item())
        for p in range(int(pages_per_batch[b].item())):
            valid = min(page_size, cache_len - p * page_size)
            page_id = int(page_ids[logical_page0 + p].item())
            src0 = k0 + p * page_size
            src1 = src0 + valid
            kv_cache[page_id, 0, :valid] = k[src0:src1]
            kv_cache[page_id, 1, :valid] = v[src0:src1]
    return kv_cache, page_offsets, page_ids, last_page_lens


def quantize_paged_kv_inputs(inputs: dict, headdim: int):
    q_raw = inputs["q"]
    k_raw = inputs["k"]
    v_raw = inputs["v"]
    cu_q = inputs["cu_q"]
    page_size = fp8_block_n(headdim, inputs["case"].has_rab)
    if page_size != 64:
        raise ValueError(f"Blackwell RTX FP8 paged KV benchmark requires page_size=64, got {page_size}")
    kv_cache_raw, page_offsets, page_ids, last_page_lens = make_paged_cache_from_varlen_kv(
        inputs, page_size
    )
    q_raw = fp8_round_bf16(q_raw)
    k_raw = fp8_round_bf16(k_raw)
    v_raw = fp8_round_bf16(v_raw)
    kv_cache_raw = fp8_round_bf16(kv_cache_raw)

    # Q/K: block-scale along D.
    q_fp8, q_descale, cu_q_blk = quantize_for_block_scale(
        q_raw, cu_q, fp8_type=torch.float8_e4m3fn,
        scale_mode="token_dchunk", d_chunk_size=128, round_to_e8m0=True
    )
    k_fp8, k_descale, cu_k_blk = quantize_for_block_scale(
        k_raw, cu_q, fp8_type=torch.float8_e4m3fn,
        scale_mode="token_dchunk", d_chunk_size=128, round_to_e8m0=True
    )
    # V target tensor: block-scale along N.  It is unused when target_len=0,
    # but the kernel path expects valid scale metadata.
    v_fp8, v_descale, cu_v_blk = quantize_for_block_scale(
        v_raw, cu_q, block_size=page_size, fp8_type=torch.float8_e4m3fn,
        scale_mode="seq_block", round_to_e8m0=True
    )

    sf_q = pack_descale_to_e8m0x4_int32(q_descale)
    sf_k = pack_descale_to_e8m0x4_int32(k_descale)
    sf_v = pack_descale_to_e8m0x4_int32(v_descale).repeat_interleave(page_size, dim=1)

    kv_cache_fp8, sf_k_cache, sf_v_cache = quantize_paged_kv_cache_for_block_scale(
        kv_cache_raw, block_size=page_size, fp8_type=torch.float8_e4m3fn
    )
    sf_k = torch.cat([sf_k_cache, sf_k], dim=1).contiguous()
    sf_v = torch.cat([sf_v_cache, sf_v], dim=1).contiguous()

    return (
        q_fp8, k_fp8, v_fp8, kv_cache_fp8,
        sf_q, sf_k, sf_v,
        q_descale, k_descale, v_descale,
        cu_q, inputs["cu_k"], cu_q_blk, cu_k_blk, cu_v_blk,
        inputs["num_targets"], page_offsets, page_ids, last_page_lens,
    )


def run_kernel_bf16(inputs: dict):
    """Run BF16 kernel directly via torch.ops."""
    out, _ = torch.ops.fbgemm.hstu_varlen_fwd_120(
        inputs["q"], inputs["k"], inputs["v"],
        inputs["cu_q"], inputs["cu_k"],
        inputs["seqused_q"], inputs["seqused_k"],
        inputs["max_seqlen_q"], inputs["max_seqlen_k"],
        -1,                        # scaling_seqlen: default to max_seqlen_q
        inputs["num_contexts"],
        inputs["num_targets"],
        inputs["target_group_size"],
        inputs["window_size"][0], inputs["window_size"][1],
        1.0,                       # alpha
        inputs["rab"],
        inputs["func"],
        -1,                        # quant_mode (BF16)
        None, None, None,          # descale_q/k/v
        None, None, None,          # sf_q/k/v
        None, None, None,          # cu_q_blk/kv_blk/v_blk
    )
    return out


def run_kernel_fp8bs(inputs: dict, fp8_inputs):
    """Run FP8 block-scale kernel directly via torch.ops."""
    (
        q_fp8, k_fp8, v_fp8,
        sf_q, sf_k, sf_v,
        q_descale, k_descale, v_descale,
        cu_q_blk, cu_kv_blk, cu_v_blk,
        cu_q, cu_k,
    ) = fp8_inputs
    out, _ = torch.ops.fbgemm.hstu_varlen_fwd_120(
        q_fp8, k_fp8, v_fp8,
        cu_q, cu_k,
        inputs["seqused_q"], inputs["seqused_k"],
        inputs["max_seqlen_q"], inputs["max_seqlen_k"],
        -1,
        inputs["num_contexts"], inputs["num_targets"], inputs["target_group_size"],
        inputs["window_size"][0], inputs["window_size"][1],
        1.0,
        inputs["rab"], inputs["func"],
        2,                          # quant_mode=2 (FP8 block-scale)
        q_descale, k_descale, v_descale,  # raw descales (for reference, unused by mode=2)
        sf_q, sf_k, sf_v,           # block-scale SF tensors (packed e8m0×4 int32)
        cu_q_blk, cu_kv_blk, cu_v_blk,
    )
    return out


def run_kernel_fp8bs_paged(inputs: dict, paged_inputs):
    """Run Blackwell RTX FP8 paged-KV kernel directly via torch.ops."""
    (
        q_fp8, k_fp8, v_fp8, kv_cache_fp8,
        sf_q, sf_k, sf_v,
        q_descale, k_descale, v_descale,
        cu_q, cu_k, cu_q_blk, cu_kv_blk, cu_v_blk,
        num_targets, page_offsets, page_ids, last_page_lens,
    ) = paged_inputs
    out, _ = torch.ops.fbgemm.hstu_varlen_fwd_120(
        q_fp8, k_fp8, v_fp8,
        cu_q, cu_k,
        inputs["seqused_q"], inputs["seqused_k"],
        inputs["max_seqlen_q"], inputs["max_seqlen_k"],
        -1,
        inputs["num_contexts"], num_targets, inputs["target_group_size"],
        inputs["window_size"][0], inputs["window_size"][1],
        1.0,
        inputs["rab"], inputs["func"],
        2,
        q_descale, k_descale, v_descale,
        sf_q, sf_k, sf_v,
        cu_q_blk, cu_kv_blk, cu_v_blk,
        kv_cache_fp8, page_offsets, page_ids, last_page_lens,
    )
    return out


def time_kernel(fn, warmup=WARMUP, iters=ITERS):
    """Time a CUDA kernel function; returns (avg_ms, min_ms, max_ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_evts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_evts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_evts[i].record()
        fn()
        end_evts[i].record()
    torch.cuda.synchronize()

    lats = [start_evts[i].elapsed_time(end_evts[i]) for i in range(iters)]
    return sum(lats) / len(lats), min(lats), max(lats)


# ---------------------------------------------------------------------------
# Kernel-only benchmark: measures only the CUDA attention kernel time
# ---------------------------------------------------------------------------

def bench_kernel_only(
    batch_size: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    case: BenchCase,
    columns: Sequence[str],
    rab_heads: str,
) -> dict:
    """
    Returns dict with keys: bf16_ms, fp8_ms, bf16_tflops, fp8_tflops, speedup_pct.
    Total FLOPs = 4 * valid_attention_pairs * H * D (FWD attention: GEMM1 + GEMM2).
    """
    result = {}

    try:
        inputs = make_case_inputs(batch_size, seqlen, nheads, headdim, case, rab_heads)
        result["pairs"] = inputs["pairs"]
    except Exception as e:
        result["input_error"] = str(e)
        return result

    if "fp8" in columns:
        # Prepare FP8 inputs (quantization done once, not timed)
        reason = column_unsupported_reason("fp8", headdim, case)
        if reason:
            result["fp8_unsupported"] = reason
        else:
            try:
                fp8_inputs = quantize_fp8_bs(
                    inputs["q"], inputs["k"], inputs["v"],
                    inputs["cu_q"], inputs["cu_k"], headdim, case.has_rab
                )
                fp8_avg, _, _ = time_kernel(lambda: run_kernel_fp8bs(inputs, fp8_inputs))
                result["fp8_ms"] = fp8_avg

            except Exception as e:
                result["fp8_error"] = str(e)

    if "paged" in columns:
        # Prepare paged-KV FP8 inputs once; kernel-only timing excludes
        # page/cache construction and FP8 quantization.
        reason = column_unsupported_reason("paged", headdim, case)
        if reason:
            result["paged_unsupported"] = reason
        else:
            try:
                paged_inputs = quantize_paged_kv_inputs(inputs, headdim)
                paged_avg, _, _ = time_kernel(
                    lambda: run_kernel_fp8bs_paged(inputs, paged_inputs)
                )
                result["paged_ms"] = paged_avg

            except Exception as e:
                result["paged_error"] = str(e)

    if "bf16" in columns:
        # BF16 is timed last so unsupported BF16 specializations do not prevent
        # FP8/paged measurements for the same logical case.
        reason = column_unsupported_reason("bf16", headdim, case)
        if reason:
            result["bf16_unsupported"] = reason
        else:
            try:
                bf16_avg, _, _ = time_kernel(lambda: run_kernel_bf16(inputs))
                result["bf16_ms"] = bf16_avg
            except Exception as e:
                result["bf16_error"] = str(e)

    # Compute TFLOPS
    total_flops = 4.0 * result.get("pairs", 0) * nheads * headdim
    if "bf16_ms" in result:
        result["bf16_tflops"] = total_flops / (result["bf16_ms"] * 1e-3) / 1e12
    if "fp8_ms" in result:
        result["fp8_tflops"] = total_flops / (result["fp8_ms"] * 1e-3) / 1e12
    if "paged_ms" in result:
        result["paged_tflops"] = total_flops / (result["paged_ms"] * 1e-3) / 1e12

    # Speedup
    if "bf16_tflops" in result and "fp8_tflops" in result:
        result["speedup_pct"] = (
            result["fp8_tflops"] / result["bf16_tflops"] - 1.0
        ) * 100.0
    if "fp8_ms" in result and "paged_ms" in result:
        result["paged_vs_fp8_pct"] = (
            result["fp8_ms"] / result["paged_ms"] - 1.0
        ) * 100.0

    return result


# ---------------------------------------------------------------------------
# End-to-end benchmark: includes quantization overhead
# ---------------------------------------------------------------------------

def run_e2e_bf16(inputs: dict):
    return run_kernel_bf16(inputs)


def run_e2e_fp8bs(inputs: dict, headdim: int):
    """Full end-to-end FP8: quantization + kernel."""
    fp8_inputs = quantize_fp8_bs(
        inputs["q"], inputs["k"], inputs["v"],
        inputs["cu_q"], inputs["cu_k"], headdim, inputs["case"].has_rab
    )
    return run_kernel_fp8bs(inputs, fp8_inputs)


def run_e2e_fp8bs_paged(inputs: dict, headdim: int):
    """Paged-KV e2e path: quantize Q/target tensors/cache, then run kernel."""
    paged_inputs = quantize_paged_kv_inputs(inputs, headdim)
    return run_kernel_fp8bs_paged(inputs, paged_inputs)


def bench_e2e(
    batch_size: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    case: BenchCase,
    columns: Sequence[str],
    rab_heads: str,
) -> dict:
    """Returns dict with end-to-end BF16 and FP8 latencies and speedup."""
    result = {}

    try:
        inputs = make_case_inputs(batch_size, seqlen, nheads, headdim, case, rab_heads)
        result["pairs"] = inputs["pairs"]
    except Exception as e:
        result["input_error"] = str(e)
        return result

    if "fp8" in columns:
        reason = column_unsupported_reason("fp8", headdim, case)
        if reason:
            result["fp8_unsupported"] = reason
        else:
            try:
                fp8_avg, _, _ = time_kernel(lambda: run_e2e_fp8bs(inputs, headdim))
                result["fp8_ms"] = fp8_avg

                total_flops_e2e = 4.0 * inputs["pairs"] * nheads * headdim
                result["fp8_tflops"] = total_flops_e2e / (fp8_avg * 1e-3) / 1e12

            except Exception as e:
                result["fp8_error"] = str(e)

    if "paged" in columns:
        reason = column_unsupported_reason("paged", headdim, case)
        if reason:
            result["paged_unsupported"] = reason
        else:
            try:
                paged_avg, _, _ = time_kernel(
                    lambda: run_e2e_fp8bs_paged(inputs, headdim)
                )
                result["paged_ms"] = paged_avg

                total_flops_e2e = 4.0 * inputs["pairs"] * nheads * headdim
                result["paged_tflops"] = total_flops_e2e / (paged_avg * 1e-3) / 1e12

            except Exception as e:
                result["paged_error"] = str(e)

    if "bf16" in columns:
        reason = column_unsupported_reason("bf16", headdim, case)
        if reason:
            result["bf16_unsupported"] = reason
        else:
            try:
                bf16_avg, _, _ = time_kernel(lambda: run_e2e_bf16(inputs))
                result["bf16_ms"] = bf16_avg

                total_flops_e2e = 4.0 * inputs["pairs"] * nheads * headdim
                result["bf16_tflops"] = total_flops_e2e / (bf16_avg * 1e-3) / 1e12

            except Exception as e:
                result["bf16_error"] = str(e)

    if "bf16_ms" in result and "fp8_ms" in result:
        result["speedup_pct"] = (
            result["fp8_ms"] / result["bf16_ms"] - 1.0
        ) * -100.0  # positive = FP8 faster
    if "fp8_ms" in result and "paged_ms" in result:
        result["paged_vs_fp8_pct"] = (
            result["fp8_ms"] / result["paged_ms"] - 1.0
        ) * 100.0

    return result


# ---------------------------------------------------------------------------
# Pretty print helpers
# ---------------------------------------------------------------------------

def _case_config_label(bs: int, seqlen: int, nheads: int, headdim: int,
                       case: BenchCase, rab_heads: str) -> str:
    suffix = " rab_h=1" if case.has_rab and rab_heads == "shared" else ""
    return f"bs={bs} seq={seqlen} h={nheads} d={headdim} {case.label}{suffix}"


def print_kernel_table(
    batch_sizes, seqlens, nheads_list, headdims, cases, columns, rab_heads
):
    print("=" * 120)
    print("KERNEL-ONLY BENCHMARK  (FP8 quantization not included)")
    print("=" * 120)
    hdr = (f"{'Config':<72} {'BF16(ms)':>9} {'BF16 TFLOPS':>12}"
           f" {'FP8(ms)':>9} {'FP8 TFLOPS':>12}"
           f" {'Paged(ms)':>10} {'Paged TFLOPS':>13} {'Paged/F8':>9}"
           f" {'Speedup':>9}")
    print(hdr)
    print("-" * len(hdr))

    for bs in batch_sizes:
        for seqlen in seqlens:
            for nheads in nheads_list:
                for headdim in headdims:
                    for case in cases:
                        if not has_supported_column(columns, headdim, case):
                            continue
                        cfg = _case_config_label(bs, seqlen, nheads, headdim, case, rab_heads)
                        res = bench_kernel_only(bs, seqlen, nheads, headdim, case, columns, rab_heads)

                        bf16_str = (f"{res['bf16_ms']:9.3f}" if "bf16_ms" in res
                                    else (f"{'N/A':>9}" if "bf16" not in columns or "bf16_unsupported" in res else f"{'ERR':>9}"))
                        bf16_tf = (f"{res['bf16_tflops']:12.1f}" if "bf16_tflops" in res
                                   else (f"{'N/A':>12}" if "bf16" not in columns or "bf16_unsupported" in res else f"{'ERR':>12}"))
                        fp8_str = (f"{res['fp8_ms']:9.3f}" if "fp8_ms" in res
                                   else (f"{'N/A':>9}" if "fp8" not in columns or "fp8_unsupported" in res else f"{'ERR':>9}"))
                        fp8_tf = (f"{res['fp8_tflops']:12.1f}" if "fp8_tflops" in res
                                  else (f"{'N/A':>12}" if "fp8" not in columns or "fp8_unsupported" in res else f"{'ERR':>12}"))
                        paged_str = (f"{res['paged_ms']:10.3f}" if "paged_ms" in res
                                     else (f"{'N/A':>10}" if "paged" not in columns or "paged_unsupported" in res else f"{'ERR':>10}"))
                        paged_tf = (f"{res['paged_tflops']:13.1f}" if "paged_tflops" in res
                                    else (f"{'N/A':>13}" if "paged" not in columns or "paged_unsupported" in res else f"{'ERR':>13}"))
                        paged_vs = (f"{res['paged_vs_fp8_pct']:+8.1f}%"
                                    if "paged_vs_fp8_pct" in res
                                    else f"{'N/A':>9}")
                        speedup = (f"{res['speedup_pct']:+8.1f}%"
                                   if "speedup_pct" in res
                                   else f"{'N/A':>9}")

                        print(
                            f"  {cfg:<70} {bf16_str} {bf16_tf} {fp8_str} {fp8_tf}"
                            f" {paged_str} {paged_tf} {paged_vs} {speedup}"
                        )

                        if "bf16_error" in res:
                            print(f"    BF16 ERROR: {res['bf16_error']}")
                        if "fp8_error" in res:
                            print(f"    FP8  ERROR: {res['fp8_error']}")
                        if "paged_error" in res:
                            print(f"    PAGED ERROR: {res['paged_error']}")
                        if "input_error" in res:
                            print(f"    INPUT ERROR: {res['input_error']}")
    print()


def print_e2e_table(
    batch_sizes, seqlens, nheads_list, headdims, cases, columns, rab_heads
):
    print("=" * 120)
    print("END-TO-END BENCHMARK  (includes FP8 quantization overhead)")
    print("=" * 120)
    hdr = (f"{'Config':<72} {'BF16(ms)':>9} {'BF16 TFLOPS':>12}"
           f" {'FP8 e2e(ms)':>11} {'FP8 TFLOPS':>12}"
           f" {'Paged(ms)':>10} {'Paged TFLOPS':>13} {'Paged/F8':>9}"
           f" {'Speedup':>9}")
    print(hdr)
    print("-" * len(hdr))

    for bs in batch_sizes:
        for seqlen in seqlens:
            for nheads in nheads_list:
                for headdim in headdims:
                    for case in cases:
                        if not has_supported_column(columns, headdim, case):
                            continue
                        cfg = _case_config_label(bs, seqlen, nheads, headdim, case, rab_heads)
                        res = bench_e2e(bs, seqlen, nheads, headdim, case, columns, rab_heads)

                        bf16_str = (f"{res['bf16_ms']:9.3f}" if "bf16_ms" in res
                                    else (f"{'N/A':>9}" if "bf16" not in columns or "bf16_unsupported" in res else f"{'ERR':>9}"))
                        bf16_tf = (f"{res['bf16_tflops']:12.1f}" if "bf16_tflops" in res
                                   else (f"{'N/A':>12}" if "bf16" not in columns or "bf16_unsupported" in res else f"{'ERR':>12}"))
                        fp8_str = (f"{res['fp8_ms']:11.3f}" if "fp8_ms" in res
                                   else (f"{'N/A':>11}" if "fp8" not in columns or "fp8_unsupported" in res else f"{'ERR':>11}"))
                        fp8_tf = (f"{res['fp8_tflops']:12.1f}" if "fp8_tflops" in res
                                  else (f"{'N/A':>12}" if "fp8" not in columns or "fp8_unsupported" in res else f"{'ERR':>12}"))
                        paged_str = (f"{res['paged_ms']:10.3f}" if "paged_ms" in res
                                     else (f"{'N/A':>10}" if "paged" not in columns or "paged_unsupported" in res else f"{'ERR':>10}"))
                        paged_tf = (f"{res['paged_tflops']:13.1f}" if "paged_tflops" in res
                                    else (f"{'N/A':>13}" if "paged" not in columns or "paged_unsupported" in res else f"{'ERR':>13}"))
                        paged_vs = (f"{res['paged_vs_fp8_pct']:+8.1f}%"
                                    if "paged_vs_fp8_pct" in res
                                    else f"{'N/A':>9}")
                        speedup_raw = res.get("speedup_pct")
                        if speedup_raw is not None:
                            speedup = f"{speedup_raw:+8.1f}%"
                        else:
                            speedup = f"{'N/A':>9}"

                        print(
                            f"  {cfg:<70} {bf16_str} {bf16_tf} {fp8_str} {fp8_tf}"
                            f" {paged_str} {paged_tf} {paged_vs} {speedup}"
                        )

                        if "bf16_error" in res:
                            print(f"    BF16 ERROR: {res['bf16_error']}")
                        if "fp8_error" in res:
                            print(f"    FP8  ERROR: {res['fp8_error']}")
                        if "paged_error" in res:
                            print(f"    PAGED ERROR: {res['paged_error']}")
                        if "input_error" in res:
                            print(f"    INPUT ERROR: {res['input_error']}")
    print()


# ---------------------------------------------------------------------------
# Accuracy check
# ---------------------------------------------------------------------------

def check_accuracy(
    batch_sizes, seqlens, nheads_list, headdims, cases, rab_heads
):
    print("=" * 120)
    print("ACCURACY CHECK  (FP8 quant_mode=2 vs BF16)")
    print("=" * 120)
    hdr = f"{'Config':<72} {'max_err':>10} {'mean_err':>10} {'cos_sim':>10} {'PASS?':>6}"
    print(hdr)
    print("-" * len(hdr))

    all_pass = True
    skipped = 0
    for bs in batch_sizes:
        for seqlen in seqlens:
            for nheads in nheads_list:
                for headdim in headdims:
                    for case in cases:
                        cfg = _case_config_label(bs, seqlen, nheads, headdim, case, rab_heads)
                        try:
                            inputs = make_case_inputs(bs, seqlen, nheads, headdim, case, rab_heads)
                            bf16_reason = column_unsupported_reason("bf16", headdim, case)
                            fp8_reason = column_unsupported_reason("fp8", headdim, case)
                            if bf16_reason or fp8_reason:
                                skipped += 1
                                continue
                            # BF16 reference
                            ref = run_kernel_bf16(inputs).float()

                            # FP8 output
                            fp8_inputs = quantize_fp8_bs(
                                inputs["q"], inputs["k"], inputs["v"],
                                inputs["cu_q"], inputs["cu_k"], headdim, case.has_rab
                            )
                            fp8_out = run_kernel_fp8bs(inputs, fp8_inputs).float()

                            max_err = (fp8_out - ref).abs().max().item()
                            mean_err = (fp8_out - ref).abs().mean().item()
                            cos_sim = F.cosine_similarity(
                                fp8_out.flatten().unsqueeze(0),
                                ref.flatten().unsqueeze(0),
                            ).item()
                            passed = cos_sim > 0.99
                            if not passed:
                                all_pass = False
                            status = "PASS" if passed else "FAIL"
                            print(f"  {cfg:<70} {max_err:10.5f} {mean_err:10.5f} {cos_sim:10.6f} {status:>6}")
                        except Exception as e:
                            all_pass = False
                            print(f"  {cfg:<70} ERROR: {e}")
    print()
    if all_pass:
        print("All accuracy checks PASSED.")
    else:
        print("Some accuracy checks FAILED!")
    if skipped:
        print(f"Skipped {skipped} unsupported accuracy configs listed in the startup summary.")
    print()
    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SM120 FP8 HSTU kernel-only benchmark."
    )
    parser.add_argument(
        "--mode",
        choices=["kernel", "e2e", "accuracy", "all"],
        default="all",
        help="Which benchmark to run (default: all)",
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 4, 8],
        metavar="BS",
    )
    parser.add_argument(
        "--seqlens", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096],
        metavar="S",
    )
    parser.add_argument(
        "--nheads", type=int, nargs="+", default=[4, 16],
        metavar="H",
    )
    parser.add_argument(
        "--headdims", type=int, nargs="+", default=[32, 64, 128, 256],
        metavar="D",
    )
    parser.add_argument(
        "--causal-only", action="store_true",
        help="Only benchmark causal attention (alias for --mask-configs causal)",
    )
    parser.add_argument(
        "--full-only", action="store_true",
        help="Only benchmark full attention (alias for --mask-configs full)",
    )
    parser.add_argument(
        "--mask-configs",
        nargs="+",
        default=["all"],
        metavar="MASK",
        help=(
            "Mask configs to benchmark. Use 'all' or any of: "
            f"{', '.join(MASK_CONFIGS)}. Default: all"
        ),
    )
    parser.add_argument(
        "--bias-configs",
        nargs="+",
        default=["all"],
        metavar="BIAS",
        help=(
            "Bias configs to benchmark. Use 'all' or any of: "
            f"{', '.join(BIAS_CONFIGS)}. Default: all"
        ),
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["all"],
        metavar="COL",
        help=(
            "Output columns to time. Use 'all' or any of: "
            f"{', '.join(COLUMNS)}. Default: all"
        ),
    )
    parser.add_argument(
        "--rab-heads",
        choices=RAB_HEAD_MODES,
        default="per-head",
        help=(
            "RAB head layout for generated inputs: per-head uses heads_rab=heads; "
            "shared uses heads_rab=1 for RAB/DRAB cases. Default: per-head"
        ),
    )
    parser.add_argument(
        "--warmup", type=int, default=WARMUP,
        help=f"Number of warmup iterations (default: {WARMUP})",
    )
    parser.add_argument(
        "--iters", type=int, default=ITERS,
        help=f"Number of timed iterations (default: {ITERS})",
    )
    return parser.parse_args()


def _expand_choices(name: str, values: Sequence[str], allowed: Sequence[str]) -> List[str]:
    if "all" in values:
        if len(values) > 1:
            raise ValueError(f"--{name} cannot combine 'all' with explicit values")
        return list(allowed)
    bad = [v for v in values if v not in allowed]
    if bad:
        raise ValueError(f"invalid --{name} values {bad}; allowed: {list(allowed)}")
    return list(values)


def main():
    args = parse_args()
    check_blackwell_rtx()

    global WARMUP, ITERS
    WARMUP = args.warmup
    ITERS = args.iters

    if args.causal_only and args.full_only:
        raise ValueError("--causal-only and --full-only are mutually exclusive")
    if args.causal_only:
        mask_configs = ["causal"]
    elif args.full_only:
        mask_configs = ["full"]
    else:
        mask_configs = _expand_choices("mask-configs", args.mask_configs, MASK_CONFIGS)
    bias_configs = _expand_choices("bias-configs", args.bias_configs, BIAS_CONFIGS)
    columns = _expand_choices("columns", args.columns, COLUMNS)
    cases = build_cases(mask_configs, bias_configs)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Capability: SM{torch.cuda.get_device_capability()[0]}"
          f"{torch.cuda.get_device_capability()[1]}")
    print(f"Warmup={WARMUP} Iters={ITERS}")
    print(f"Mask configs={mask_configs}")
    print(f"Bias configs={bias_configs}")
    print(f"Columns={columns}")
    print(f"RAB heads={args.rab_heads}")
    print(f"Total logical configs={len(cases)}")
    print()
    print_unsupported_summary(args.headdims, cases, columns)

    if args.mode in ("accuracy", "all"):
        # Use smaller config for accuracy check to save time
        check_accuracy(
            batch_sizes=[1],
            seqlens=[128, 256, 512],
            nheads_list=[1, 4],
            headdims=args.headdims,
            cases=cases,
            rab_heads=args.rab_heads,
        )

    if args.mode in ("kernel", "all"):
        print_kernel_table(
            batch_sizes=args.batch_sizes,
            seqlens=args.seqlens,
            nheads_list=args.nheads,
            headdims=args.headdims,
            cases=cases,
            columns=columns,
            rab_heads=args.rab_heads,
        )

    if args.mode in ("e2e", "all"):
        print_e2e_table(
            batch_sizes=args.batch_sizes,
            seqlens=args.seqlens,
            nheads_list=args.nheads,
            headdims=args.headdims,
            cases=cases,
            columns=columns,
            rab_heads=args.rab_heads,
        )


if __name__ == "__main__":
    main()
