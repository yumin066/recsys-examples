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
ENV_FILE="${HSTU_L40S_ENV_FILE:-${HSTU_BUILD_CACHE}/l40s_bf16_kernel_only_env.sh}"

if [[ "${HSTU_SKIP_SETUP:-0}" != "1" ]]; then
  "${SCRIPT_DIR}/setup_l40s_bf16_kernel_only_env.sh"
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing env file: ${ENV_FILE}" >&2
  echo "Run setup_l40s_bf16_kernel_only_env.sh first." >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

if [[ "$#" -eq 0 ]]; then
  set -- --output-csv "${HSTU_BUILD_CACHE}/l40s_bf16_kernel_only.csv"
fi

python "${SCRIPT_DIR}/l40s_bf16_kernel_only.py" "$@"
