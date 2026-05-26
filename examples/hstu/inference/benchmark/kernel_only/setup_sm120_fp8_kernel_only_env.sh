#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd -P)"

resolve_build_cache() {
  if [[ -n "${HSTU_BUILD_CACHE:-}" ]]; then
    printf '%s\n' "${HSTU_BUILD_CACHE}"
  else
    printf '%s\n' "${SCRIPT_DIR}/build_cache"
  fi
}

resolve_fbgemm_hstu_path() {
  if [[ -n "${FBGEMM_HSTU_PATH:-}" ]]; then
    printf '%s\n' "${FBGEMM_HSTU_PATH}"
  elif [[ -d "${REPO_ROOT}/third_party/FBGEMM/fbgemm_gpu/experimental/hstu" ]]; then
    printf '%s\n' "${REPO_ROOT}/third_party/FBGEMM/fbgemm_gpu/experimental/hstu"
  elif [[ -d "${REPO_ROOT}/../fbgemm-hstu/fbgemm_gpu/experimental/hstu" ]]; then
    printf '%s\n' "${REPO_ROOT}/../fbgemm-hstu/fbgemm_gpu/experimental/hstu"
  else
    echo "Set FBGEMM_HSTU_PATH to <fbgemm-hstu>/fbgemm_gpu/experimental/hstu." >&2
    exit 2
  fi
}

HSTU_BUILD_CACHE="$(resolve_build_cache)"
mkdir -p "${HSTU_BUILD_CACHE}"
HSTU_BUILD_CACHE="$(cd "${HSTU_BUILD_CACHE}" && pwd -P)"
FBGEMM_HSTU_PATH="$(resolve_fbgemm_hstu_path)"
FBGEMM_HSTU_BUILD_LIB="${FBGEMM_HSTU_BUILD_LIB:-${HSTU_BUILD_CACHE}/fbgemm_hstu_build/lib}"
FBGEMM_HSTU_BUILD_TEMP="${FBGEMM_HSTU_BUILD_TEMP:-${HSTU_BUILD_CACHE}/fbgemm_hstu_build/temp}"
ENV_FILE="${HSTU_SM120_ENV_FILE:-${HSTU_BUILD_CACHE}/sm120_fp8_kernel_only_env.sh}"
EXPECTED_FBGEMM_COMMIT="${EXPECTED_FBGEMM_COMMIT:-1263aa0236e1518434e5808374ad47274cfb97b6}"

mkdir -p \
  "${HSTU_BUILD_CACHE}/torch_extensions" \
  "${HSTU_BUILD_CACHE}/cuda_cache" \
  "${HSTU_BUILD_CACHE}/pip_cache" \
  "${FBGEMM_HSTU_BUILD_LIB}" \
  "${FBGEMM_HSTU_BUILD_TEMP}"

if [[ ! -f "${FBGEMM_HSTU_PATH}/hstu/cuda_hstu_attention.py" ]]; then
  echo "Missing FBGEMM HSTU Python package under FBGEMM_HSTU_PATH: ${FBGEMM_HSTU_PATH}" >&2
  exit 2
fi

actual_commit="$(git -C "${FBGEMM_HSTU_PATH}" rev-parse HEAD 2>/dev/null || true)"
if [[ -n "${actual_commit}" && "${actual_commit}" != "${EXPECTED_FBGEMM_COMMIT}" ]]; then
  echo "WARNING: expected FBGEMM commit ${EXPECTED_FBGEMM_COMMIT}, got ${actual_commit}" >&2
fi

if [[ ! -f "${FBGEMM_HSTU_BUILD_LIB}/hstu/fbgemm_gpu_experimental_hstu.so" && \
      ! -f "${FBGEMM_HSTU_PATH}/hstu/fbgemm_gpu_experimental_hstu.so" && \
      "${HSTU_SKIP_SM120_BUILD:-0}" != "1" ]]; then
  if [[ ! -f "${FBGEMM_HSTU_PATH}/../../../external/cutlass/include/cutlass/cutlass.h" ]]; then
    echo "Missing FBGEMM external/cutlass. Run: git -C third_party/FBGEMM submodule update --init external/cutlass" >&2
    exit 2
  fi

  cd "${FBGEMM_HSTU_PATH}"
  HSTU_FORCE_BUILD=TRUE \
  HSTU_DISABLE_BACKWARD=TRUE \
  HSTU_DISABLE_FP16=TRUE \
  HSTU_DISABLE_HDIM32=TRUE \
  HSTU_DISABLE_HDIM64=TRUE \
  HSTU_DISABLE_HDIM128=TRUE \
  HSTU_DISABLE_LOCAL=TRUE \
  HSTU_DISABLE_CAUSAL=TRUE \
  HSTU_DISABLE_CONTEXT=TRUE \
  HSTU_DISABLE_TARGET=TRUE \
  HSTU_DISABLE_ARBITRARY=TRUE \
  HSTU_DISABLE_RAB=TRUE \
  HSTU_DISABLE_DRAB=TRUE \
  HSTU_DISABLE_120=FALSE \
  HSTU_ARCH_LIST="${HSTU_ARCH_LIST:-12.0}" \
  MAX_JOBS="${MAX_JOBS:-4}" \
  NVCC_THREADS="${NVCC_THREADS:-2}" \
  python3 setup.py \
    build_py --build-lib "${FBGEMM_HSTU_BUILD_LIB}" \
    build_ext \
      --build-temp "${FBGEMM_HSTU_BUILD_TEMP}" \
      --build-lib "${FBGEMM_HSTU_BUILD_LIB}"
fi

export FBGEMM_HSTU_PATH
export FBGEMM_HSTU_BUILD_LIB
export ENABLE_SM120_PAGED_FP8=1
export TORCH_EXTENSIONS_DIR="${HSTU_BUILD_CACHE}/torch_extensions"
export CUDA_CACHE_PATH="${HSTU_BUILD_CACHE}/cuda_cache"
export PIP_CACHE_DIR="${HSTU_BUILD_CACHE}/pip_cache"
export PYTHONPATH="${FBGEMM_HSTU_BUILD_LIB}:${FBGEMM_HSTU_PATH}:${PYTHONPATH:-}"

python3 - <<'PY'
import os
import sys

import torch

path = os.environ["FBGEMM_HSTU_PATH"]
build_lib = os.environ.get("FBGEMM_HSTU_BUILD_LIB")
paths = []
if build_lib:
    paths.append(build_lib)
paths.append(path)
sys.path[:0] = paths

print("python_executable=", sys.executable)
print("torch_version=", torch.__version__)
print("cuda_version=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

capability = torch.cuda.get_device_capability(0)
print("device_name=", torch.cuda.get_device_name(0))
print("device_capability=", capability)
if capability[0] < 12:
    raise RuntimeError(f"SM120 FP8 kernel requires SM12.x, got SM{capability[0]}{capability[1]}")

import hstu  # noqa: F401,E402

print("hstu_module=", getattr(hstu, "__file__", "<builtin>"))
print("has_hstu_varlen_fwd_120=", hasattr(torch.ops.fbgemm, "hstu_varlen_fwd_120"))
if not hasattr(torch.ops.fbgemm, "hstu_varlen_fwd_120"):
    raise RuntimeError("torch.ops.fbgemm.hstu_varlen_fwd_120 is not registered")
PY

cat >"${ENV_FILE}" <<EOF
export HSTU_BUILD_CACHE="${HSTU_BUILD_CACHE}"
export TORCH_EXTENSIONS_DIR="${HSTU_BUILD_CACHE}/torch_extensions"
export CUDA_CACHE_PATH="${HSTU_BUILD_CACHE}/cuda_cache"
export PIP_CACHE_DIR="${HSTU_BUILD_CACHE}/pip_cache"
export FBGEMM_HSTU_PATH="${FBGEMM_HSTU_PATH}"
export FBGEMM_HSTU_BUILD_LIB="${FBGEMM_HSTU_BUILD_LIB}"
export EXPECTED_FBGEMM_COMMIT="${EXPECTED_FBGEMM_COMMIT}"
export ENABLE_SM120_PAGED_FP8=1
export PYTHONPATH="${FBGEMM_HSTU_BUILD_LIB}:${FBGEMM_HSTU_PATH}:\${PYTHONPATH:-}"
EOF

echo "wrote_env=${ENV_FILE}"
echo "fbgemm_hstu_path=${FBGEMM_HSTU_PATH}"
echo "fbgemm_hstu_build_lib=${FBGEMM_HSTU_BUILD_LIB}"
echo "expected_fbgemm_commit=${EXPECTED_FBGEMM_COMMIT}"
