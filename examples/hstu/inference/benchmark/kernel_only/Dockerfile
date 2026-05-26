# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.02-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update -y --fix-missing && \
    apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      ninja-build \
      pkg-config \
      rsync

RUN python3 -m pip install --no-cache-dir \
      ninja \
      packaging \
      psutil \
      setuptools \
      wheel

WORKDIR /workspace/recsys-examples
