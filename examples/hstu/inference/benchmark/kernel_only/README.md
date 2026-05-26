# HSTU Kernel-Only 性能复现

本目录用于在客户的 L40S 和 SM120 host 机器上复现 HSTU attention kernel-only 性能对比。默认假设客户已经进入本目录执行脚本：

```bash
cd examples/hstu/inference/benchmark/kernel_only
```

测试只覆盖 CUDA attention kernel 本体，不包含 recsys 端到端模型、数据加载、wrapper 量化或 paged cache 上层调度。

默认测试矩阵：

- batch size：`1 2 4 8`
- sequence length：`128 2048 4096`
- heads：`4`
- head dim：`256`
- mask：`full`
- bias：`none`
- warmup：`10`
- iters：`50`

## 通用要求

- 在对应 GPU host 上运行脚本，不要在登录节点上运行。
- 所有编译和 cache 默认写入当前目录的 `build_cache/`，不放 Docker 或 job 的 `/tmp`。
- 如需反复测试，先保留同一节点的 interactive allocation，再多次运行脚本。
- 若源码 checkout 没有 submodule 内容，请先执行：

```bash
git -C ../../../../.. submodule update --init third_party/cutlass third_party/FBGEMM
git -C ../../../../../third_party/FBGEMM submodule update --init external/cutlass
```

脚本会自动使用：

```bash
export HSTU_BUILD_CACHE="${PWD}/build_cache"
```

## Docker 复现

推荐客户使用本目录的最小 Docker 入口复现。镜像默认基于 `nvcr.io/nvidia/pytorch:26.02-py3`，可通过 `HSTU_KERNEL_ONLY_BASE_IMAGE` 覆盖；源码和 `build_cache/` 都以 volume 形式挂载进容器，编译产物仍写入当前目录下的 `build_cache/`。

```bash
./run_kernel_only_docker.sh build
```

L40S BF16：

```bash
mkdir -p build_cache
./run_kernel_only_docker.sh l40s \
  | tee build_cache/l40s_bf16_kernel_only.log
```

SM120 FP8：

```bash
mkdir -p build_cache
./run_kernel_only_docker.sh sm120 \
  | tee build_cache/sm120_fp8_kernel_only.log
```

如果已经 build 过镜像，后续复测可设置 `HSTU_SKIP_DOCKER_BUILD=1`，并复用同一个 interactive allocation 与同一个 `build_cache/`。

## L40S BF16

L40S 侧使用当前 repo 的 `corelib/hstu` BF16 kernel，并构建 `sm_89` cubin。

```bash
mkdir -p build_cache
./run_l40s_bf16_kernel_only.sh \
  | tee build_cache/l40s_bf16_kernel_only.log
```

## SM120 FP8

SM120 侧使用 `third_party/FBGEMM` pin 住的 fbgemm-hstu FP8 kernel。该 submodule 在本 commit 中固定到：

```text
1263aa0236e1518434e5808374ad47274cfb97b6
```

默认 `FBGEMM_HSTU_PATH` 为：

```text
third_party/FBGEMM/fbgemm_gpu/experimental/hstu
```

如果客户已经在 host 上另行构建了同一 commit 的 fbgemm-hstu，也可以显式设置 `FBGEMM_HSTU_PATH` 指向已构建好的 `fbgemm_gpu/experimental/hstu` 目录。

如果 host 上还没有可加载的 FBGEMM HSTU `.so`，setup 脚本会默认在 `build_cache/` 下构建一份 build/lib，并让 benchmark 优先加载这个目录。等价的手工构建命令如下：

```bash
export REPO_ROOT="$(cd ../../../../.. && pwd -P)"
export HSTU_BUILD_CACHE="${PWD}/build_cache"
export FBGEMM_HSTU_PATH="${REPO_ROOT}/third_party/FBGEMM/fbgemm_gpu/experimental/hstu"
export FBGEMM_HSTU_BUILD_LIB="${HSTU_BUILD_CACHE}/fbgemm_hstu_build/lib"

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
    --build-temp "${HSTU_BUILD_CACHE}/fbgemm_hstu_build/temp" \
    --build-lib "${FBGEMM_HSTU_BUILD_LIB}"
```

```bash
mkdir -p build_cache
./run_sm120_fp8_kernel_only.sh \
  | tee build_cache/sm120_fp8_kernel_only.log
```

SM120 脚本默认执行本目录内的 `sm120_fp8_kernel_only.py`。该文件保留了实际验证过的 SM120 FP8 benchmark 逻辑，只把 FBGEMM HSTU import path 改为 `FBGEMM_HSTU_PATH`。默认参数等价于：

```bash
python3 ./sm120_fp8_kernel_only.py \
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
```

## 对比口径

L40S 输出表中的 `BF16(ms)` 和 SM120 输出表中的 `FP8(ms)` 是 latency 对比口径。speedup 计算方式：

```text
SM120/L40S speedup = L40S BF16 latency ms / SM120 FP8 latency ms
```

在本项目验证过的代表性结果中，`bs=8 seq=4096 h=4 d=256 full` 为：

- L40S BF16：`2.712 ms`
- SM120 FP8：`0.866 ms`
- speedup：`3.13x`
