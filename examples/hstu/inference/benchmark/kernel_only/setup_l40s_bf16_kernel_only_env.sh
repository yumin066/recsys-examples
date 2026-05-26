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

HSTU_BUILD_CACHE="$(resolve_build_cache)"
mkdir -p "${HSTU_BUILD_CACHE}"
HSTU_BUILD_CACHE="$(cd "${HSTU_BUILD_CACHE}" && pwd -P)"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
SOURCE_ID="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || printf 'nogit')"
BUILD_SRC_ROOT="${BUILD_SRC_ROOT:-${HSTU_BUILD_CACHE}/src/recsys-examples-${SOURCE_ID}-l40s-bf16-sm89}"
VENV_DIR="${HSTU_KERNEL_ONLY_VENV:-${HSTU_BUILD_CACHE}/venv}"
ENV_FILE="${HSTU_L40S_ENV_FILE:-${HSTU_BUILD_CACHE}/l40s_bf16_kernel_only_env.sh}"

mkdir -p \
  "${HSTU_BUILD_CACHE}/torch_extensions" \
  "${HSTU_BUILD_CACHE}/cuda_cache" \
  "${HSTU_BUILD_CACHE}/pip_cache" \
  "${BUILD_SRC_ROOT}/corelib" \
  "${BUILD_SRC_ROOT}/third_party"

export TORCH_EXTENSIONS_DIR="${HSTU_BUILD_CACHE}/torch_extensions"
export CUDA_CACHE_PATH="${HSTU_BUILD_CACHE}/cuda_cache"
export PIP_CACHE_DIR="${HSTU_BUILD_CACHE}/pip_cache"
export HSTU_BUILD_VERSION="${SOURCE_ID}"

if [[ ! -f "${REPO_ROOT}/third_party/cutlass/include/cutlass/cutlass.h" ]]; then
  echo "Missing third_party/cutlass. Run: git submodule update --init third_party/cutlass" >&2
  exit 2
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv --system-site-packages "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python - <<'PY'
import sys
import torch

print("python_executable=", sys.executable)
print("torch_version=", torch.__version__)
print("cuda_version=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_name=", torch.cuda.get_device_name(0))
    print("device_capability=", torch.cuda.get_device_capability(0))
PY

python - <<'PY' >/dev/null 2>&1 || python -m pip install packaging ninja setuptools wheel
import packaging  # noqa: F401
import setuptools  # noqa: F401
PY

if command -v rsync >/dev/null 2>&1; then
  rsync -a --exclude build --exclude '*.egg-info' \
    "${REPO_ROOT}/corelib/hstu/" "${BUILD_SRC_ROOT}/corelib/hstu/"
  rsync -a "${REPO_ROOT}/third_party/cutlass/" "${BUILD_SRC_ROOT}/third_party/cutlass/"
else
  mkdir -p "${BUILD_SRC_ROOT}/corelib/hstu" "${BUILD_SRC_ROOT}/third_party/cutlass"
  cp -a "${REPO_ROOT}/corelib/hstu/." "${BUILD_SRC_ROOT}/corelib/hstu/"
  cp -a "${REPO_ROOT}/third_party/cutlass/." "${BUILD_SRC_ROOT}/third_party/cutlass/"
fi

cd "${BUILD_SRC_ROOT}/corelib/hstu"
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
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
python setup.py install

cat >"${ENV_FILE}" <<EOF
source "${VENV_DIR}/bin/activate"
export HSTU_BUILD_CACHE="${HSTU_BUILD_CACHE}"
export TORCH_EXTENSIONS_DIR="${HSTU_BUILD_CACHE}/torch_extensions"
export CUDA_CACHE_PATH="${HSTU_BUILD_CACHE}/cuda_cache"
export PIP_CACHE_DIR="${HSTU_BUILD_CACHE}/pip_cache"
export BUILD_SRC_ROOT="${BUILD_SRC_ROOT}"
export PYTHONPATH="${REPO_ROOT}/examples:${REPO_ROOT}/examples/hstu:\${PYTHONPATH:-}"
EOF

echo "wrote_env=${ENV_FILE}"
echo "build_src_root=${BUILD_SRC_ROOT}"
