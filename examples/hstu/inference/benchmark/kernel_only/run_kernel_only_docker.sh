#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  run_kernel_only_docker.sh build
  run_kernel_only_docker.sh l40s [benchmark args...]
  run_kernel_only_docker.sh sm120 [benchmark args...]
  run_kernel_only_docker.sh bash [bash args...]

Environment:
  HSTU_BUILD_CACHE              Build/cache directory, default ./build_cache.
  HSTU_KERNEL_ONLY_IMAGE        Docker image tag to build/run.
  HSTU_KERNEL_ONLY_BASE_IMAGE   Base image, default nvcr.io/nvidia/pytorch:26.02-py3.
  HSTU_SKIP_DOCKER_BUILD=1      Reuse an existing image tag.
EOF
}

if [[ "$#" -lt 1 ]]; then
  usage
  exit 2
fi

MODE="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd -P)"
CURRENT_USER="${USER:-$(id -un)}"
CURRENT_UID="$(id -u)"
CURRENT_GID="$(id -g)"
IMAGE="${HSTU_KERNEL_ONLY_IMAGE:-recsys-hstu-kernel-only:cuda13.1}"
BASE_IMAGE="${HSTU_KERNEL_ONLY_BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.02-py3}"

build_image() {
  docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    -t "${IMAGE}" \
    "${SCRIPT_DIR}"
}

case "${MODE}" in
  build)
    build_image
    exit 0
    ;;
  l40s|sm120|bash)
    ;;
  *)
    usage
    exit 2
    ;;
esac

resolve_build_cache() {
  if [[ -n "${HSTU_BUILD_CACHE:-}" ]]; then
    printf '%s\n' "${HSTU_BUILD_CACHE}"
  else
    printf '%s\n' "${SCRIPT_DIR}/build_cache"
  fi
}

HSTU_BUILD_CACHE="$(resolve_build_cache)"
mkdir -p "${HSTU_BUILD_CACHE}/home" "${HSTU_BUILD_CACHE}/logs"
HSTU_BUILD_CACHE="$(cd "${HSTU_BUILD_CACHE}" && pwd -P)"

if [[ "${HSTU_SKIP_DOCKER_BUILD:-0}" != "1" ]]; then
  build_image
fi

docker_args=(
  run
  --rm
  --gpus all
  --ipc=host
  --network=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  --user "${CURRENT_UID}:${CURRENT_GID}"
  -e "USER=${CURRENT_USER}"
  -e "HOME=${HSTU_BUILD_CACHE}/home"
  -e "HSTU_BUILD_CACHE=${HSTU_BUILD_CACHE}"
  -e "TORCH_EXTENSIONS_DIR=${HSTU_BUILD_CACHE}/torch_extensions"
  -e "CUDA_CACHE_PATH=${HSTU_BUILD_CACHE}/cuda_cache"
  -e "PIP_CACHE_DIR=${HSTU_BUILD_CACHE}/pip_cache"
  -e "PYTHONDONTWRITEBYTECODE=1"
  -v "${REPO_ROOT}:${REPO_ROOT}"
  -v "${HSTU_BUILD_CACHE}:${HSTU_BUILD_CACHE}"
  -w "${SCRIPT_DIR}"
)

case "${MODE}" in
  l40s)
    docker_args+=(
      -e "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}"
    )
    container_cmd=(
      bash
      "./run_l40s_bf16_kernel_only.sh"
      "$@"
    )
    ;;
  sm120)
    docker_args+=(
      -e "ENABLE_SM120_PAGED_FP8=1"
      -e "FBGEMM_HSTU_PATH=${REPO_ROOT}/third_party/FBGEMM/fbgemm_gpu/experimental/hstu"
      -e "FBGEMM_HSTU_BUILD_LIB=${HSTU_BUILD_CACHE}/fbgemm_hstu_build/lib"
      -e "HSTU_ARCH_LIST=${HSTU_ARCH_LIST:-12.0}"
      -e "MAX_JOBS=${MAX_JOBS:-4}"
      -e "NVCC_THREADS=${NVCC_THREADS:-2}"
    )
    container_cmd=(
      bash
      "./run_sm120_fp8_kernel_only.sh"
      "$@"
    )
    ;;
  bash)
    container_cmd=(bash "$@")
    ;;
esac

if [[ -n "${HSTU_DOCKER_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${HSTU_DOCKER_EXTRA_ARGS})
  docker_args+=("${extra_args[@]}")
fi

exec docker "${docker_args[@]}" "${IMAGE}" "${container_cmd[@]}"
