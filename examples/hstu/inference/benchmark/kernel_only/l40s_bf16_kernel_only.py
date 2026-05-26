# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv
import importlib
import sys
from pathlib import Path
from typing import Callable, Iterable

import torch
from hstu_attn import hstu_attn_varlen_func


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="L40S BF16 HSTU attention kernel-only benchmark."
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--seqlens", nargs="+", type=int, default=[128, 2048, 4096])
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--headdim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def module_path(name: str) -> str:
    try:
        return str(getattr(importlib.import_module(name), "__file__", "<builtin>"))
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def time_kernel(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    latencies = [start_events[i].elapsed_time(end_events[i]) for i in range(iters)]
    return sum(latencies) / len(latencies), min(latencies), max(latencies)


def make_inputs(
    *,
    batch_size: int,
    seqlen: int,
    nheads: int,
    headdim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total = batch_size * seqlen
    q = torch.randn((total, nheads, headdim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((total, nheads, headdim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((total, nheads, headdim), device="cuda", dtype=torch.bfloat16)
    cu = torch.arange(
        0,
        (batch_size + 1) * seqlen,
        seqlen,
        device="cuda",
        dtype=torch.int32,
    )
    return q, k, v, cu


def run_case(
    *,
    batch_size: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    warmup: int,
    iters: int,
) -> dict[str, float | int | str]:
    q, k, v, cu = make_inputs(
        batch_size=batch_size,
        seqlen=seqlen,
        nheads=nheads,
        headdim=headdim,
    )

    def kernel() -> torch.Tensor:
        return hstu_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=seqlen,
            max_seqlen_k=seqlen,
            num_contexts=None,
            num_targets=None,
            target_group_size=1,
            window_size=(-1, -1),
            alpha=1.0,
            rab=None,
            has_drab=False,
            scaling_seqlen=seqlen,
        )

    avg_ms, min_ms, max_ms = time_kernel(kernel, warmup=warmup, iters=iters)
    total_flops = 4.0 * batch_size * seqlen * seqlen * nheads * headdim
    tflops = total_flops / (avg_ms * 1e-3) / 1e12
    return {
        "case": f"bs={batch_size} seq={seqlen} h={nheads} d={headdim} full",
        "batch_size": batch_size,
        "seqlen": seqlen,
        "nheads": nheads,
        "headdim": headdim,
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "tflops": tflops,
    }


def write_csv(path: Path, rows: Iterable[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "batch_size",
        "seqlen",
        "nheads",
        "headdim",
        "avg_ms",
        "min_ms",
        "max_ms",
        "tflops",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    capability = torch.cuda.get_device_capability(0)
    if capability != (8, 9):
        print(
            f"WARNING: expected L40S/SM89, got SM{capability[0]}{capability[1]}",
            file=sys.stderr,
        )

    print("Device:", torch.cuda.get_device_name(0))
    print("Capability: SM%d%d" % capability)
    print("Python:", sys.executable)
    print("Torch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("hstu_attn_module:", module_path("hstu_attn"))
    print("hstu_attn_2_cuda:", module_path("hstu_attn_2_cuda"))
    print(f"Warmup={args.warmup} Iters={args.iters}")
    print("Mask configs=['full']")
    print("Bias configs=['none']")
    print("Columns=['bf16']")
    print()

    print("=" * 100)
    print("L40S BF16 KERNEL-ONLY BENCHMARK  (corelib/hstu hstu_attn_varlen_func)")
    print("=" * 100)
    header = (
        f"{'Config':<48} {'BF16(ms)':>10} {'Min(ms)':>10} "
        f"{'Max(ms)':>10} {'BF16 TFLOPS':>13}"
    )
    print(header)
    print("-" * len(header))

    rows: list[dict[str, float | int | str]] = []
    with torch.inference_mode():
        for batch_size in args.batch_sizes:
            for seqlen in args.seqlens:
                row = run_case(
                    batch_size=batch_size,
                    seqlen=seqlen,
                    nheads=args.nheads,
                    headdim=args.headdim,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                rows.append(row)
                print(
                    f"  {str(row['case']):<46} {row['avg_ms']:10.3f} "
                    f"{row['min_ms']:10.3f} {row['max_ms']:10.3f} "
                    f"{row['tflops']:13.1f}"
                )

    if args.output_csv is not None:
        write_csv(args.output_csv, rows)
        print(f"\nwrote_csv={args.output_csv}")


if __name__ == "__main__":
    main()
