#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

resolve_build_cache() {
  if [[ -n "${HSTU_BUILD_CACHE:-}" ]]; then
    printf '%s\n' "${HSTU_BUILD_CACHE}"
  else
    printf '%s\n' "${SCRIPT_DIR}/build_cache"
  fi
}

HSTU_BUILD_CACHE="$(resolve_build_cache)"
mkdir -p "${HSTU_BUILD_CACHE}"
HSTU_BUILD_CACHE="$(cd "${HSTU_BUILD_CACHE}" && pwd -P)"
ENV_FILE="${HSTU_SM120_ENV_FILE:-${HSTU_BUILD_CACHE}/sm120_fp8_kernel_only_env.sh}"

if [[ "${HSTU_SKIP_SETUP:-0}" != "1" ]]; then
  "${SCRIPT_DIR}/setup_sm120_fp8_kernel_only_env.sh"
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing env file: ${ENV_FILE}" >&2
  echo "Run setup_sm120_fp8_kernel_only_env.sh first." >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

cd "${FBGEMM_HSTU_PATH}"

if [[ "$#" -eq 0 ]]; then
  set -- \
    --mode kernel \
    --columns fp8 paged \
    --mask-configs full \
    --bias-configs none \
    --batch-sizes 1 2 4 8 \
    --seqlens 128 2048 4096 \
    --nheads 4 \
    --headdims 256 \
    --warmup 10 \
    --iters 50
fi

python3 "${SCRIPT_DIR}/sm120_fp8_kernel_only.py" "$@"
