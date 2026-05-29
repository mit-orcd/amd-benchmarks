# Megatron-LM Installation Notes — AMD MI355X

## System Info
- 8x AMD Instinct MI355X GPUs (gfx950 architecture)
- ROCm 7.2.3 at `/opt/rocm`
- System Python 3.6.8 (too old — need 3.8+)
- Working directory: `/home/v89592/shaohao/megatron-lm/`

---

## Installation Steps

### 1. Clone the ROCm fork
```bash
cd /home/v89592/shaohao
git clone https://github.com/ROCm/Megatron-LM.git megatron-lm
cd megatron-lm
```
Use the ROCm fork — it has AMD-specific patches validated on MI-series GPUs.

### 2. Get a compatible Python (3.10 recommended)
```bash
# Check if newer Python is available
python3.10 --version 2>/dev/null || python3.8 --version 2>/dev/null

# Or use conda/miniconda
conda create -n megatron python=3.10 -y
conda activate megatron
```

### 3. Install PyTorch for ROCm
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

Verify GPU is visible:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

### 4. Install Megatron-LM dependencies
```bash
pip install -r requirements.txt
```

### 5. Set ROCm environment variables
```bash
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_ROCM_ARCH=gfx950   # MI355X
```

### 6. Build fused kernels (optional, recommended for performance)
```bash
python setup.py install
```

---

## Notes for MI355X (gfx950)
- gfx950 is very new — confirm PyTorch ROCm wheel supports it before installing
- If prebuilt wheels don't support gfx950, build PyTorch from source with `PYTORCH_ROCM_ARCH=gfx950`
- Check AMD's ROCm PyTorch support matrix for the exact compatible wheel:
  https://rocm.docs.amd.com/en/latest/compatibility/pytorch-support-matrix.html

### Workaround when PyTorch lacks gfx950 kernels

The PyTorch shipped in `megatron-lm.sif` is built without gfx950 code objects
(`torch.cuda.get_arch_list()` returns `[]`). On MI355X this surfaces as:

```
RuntimeError: No HIP GPUs are available
# but torch.cuda.device_count() == 8
```

`rocminfo` will still happily list all 8 MI355X agents — the failure is
purely on the PyTorch / HIP-kernel-load side, not the device-permission side.

Fix: alias gfx950 → gfx942 at the HSA layer so the existing MI300X kernels
run on MI355X (binary-compatible at the gfx9 ISA level):

```bash
export HSA_OVERRIDE_GFX_VERSION=9.4.2
```

In `work/run.sh` this is already set inside the container env block.
Remove it once a gfx950-native PyTorch wheel is baked into the image.

---

## Container Image (Singularity) — Network Issue

Singularity is available at `/usr/bin/singularity`. However, this cluster has **no external DNS** — hub.docker.com and github.com both fail to resolve. DNS servers: `10.204.133.136` / `10.204.133.137`.

### Option 1: Ask sysadmin for internal mirror
Many HPC clusters have a local container registry or shared `.sif` files. Ask:
> "Is there a local Singularity image or container registry for ROCm/PyTorch or Megatron-LM?"

Search for existing images:
```bash
find / -maxdepth 6 -name "*.sif" 2>/dev/null
ls /shared/ /scratch/ /lustre/ /data/ /nfs/ 2>/dev/null
```

### Option 2: Download on internet-connected machine, then transfer

**If you have Singularity locally:**
```bash
singularity pull docker://rocm/megatron-lm:latest
git clone https://github.com/ROCm/Megatron-LM.git

# Transfer to cluster:
scp megatron-lm_latest.sif user@cluster:/home/v89592/shaohao/megatron-lm/
scp -r Megatron-LM/ user@cluster:/home/v89592/shaohao/megatron-lm/
```

**If you only have Docker locally:**
```bash
# On local machine:
docker pull rocm/megatron-lm:latest
docker save rocm/megatron-lm:latest -o megatron-lm.tar

# Transfer to cluster:
scp megatron-lm.tar v89592@<cluster-hostname>:/home/v89592/shaohao/megatron-lm/

# On the cluster, convert tar to .sif:
singularity build megatron-lm.sif docker-archive://megatron-lm.tar
```

#### Build fails with `no space left on device`

By default `singularity build` unpacks the rootfs into `/tmp`, which on this
node lives on `/` (~38 GB free). The image tar is ~73 GB and the unpacked
rootfs is larger still, so the build dies partway through with something like:

```
FATAL: While performing build: packer failed to pack: ...
  unpack to regular file: short write: write /tmp/build-temp-.../rootfs/...: no space left on device
```

Fix: point Singularity's tmp + cache at `/home` (705 GB free) before building.
Set both `SINGULARITY_*` and `APPTAINER_*` since `apptainer` is symlinked here:

```bash
mkdir -p /home/v89592/shaohao/singularity-tmp /home/v89592/shaohao/singularity-cache
export SINGULARITY_TMPDIR=/home/v89592/shaohao/singularity-tmp
export SINGULARITY_CACHEDIR=/home/v89592/shaohao/singularity-cache
export APPTAINER_TMPDIR=$SINGULARITY_TMPDIR
export APPTAINER_CACHEDIR=$SINGULARITY_CACHEDIR

cd /home/v89592/shaohao/megatron-lm
singularity build megatron-lm.sif docker-archive://megatron-lm.tar
```

Headroom needed is roughly **2× the tar size** (one copy for the unpacked
rootfs, one for the squashfs the builder then assembles). 73 GB tar → ~150 GB.
After the build succeeds, the tmp/cache dirs can be deleted:

```bash
rm -rf /home/v89592/shaohao/singularity-tmp /home/v89592/shaohao/singularity-cache
```

### Option 3: Use login/gateway node
This compute node may be firewalled. A login node might have outbound access — run the pull there and the `.sif` will land on shared storage.

### Running the container (once image is available)
```bash
singularity exec --rocm megatron-lm_latest.sif bash
```
The `--rocm` flag automatically exposes all AMD GPUs inside the container.

---

## Is XGMI used by Megatron-LM and `run.sh`?

Yes -- XGMI is used, but **implicitly via RCCL**, not as a direct API call.

### In `work/run.sh` -- deliberately configured to ride XGMI
- `NCCL_IB_DISABLE=1` -- disables InfiniBand (no fabric on a single node).
- `NCCL_P2P_DISABLE=0` -- **enables** GPU-to-GPU peer-to-peer, which is what
  makes RCCL pick XGMI as the transport.
- `NCCL_SHM_DISABLE=0` -- enables host-shared-memory fallback (only used
  when P2P is unavailable; not the hot path here).
- `RCCL_MSCCL_ENABLE=1` -- enables MSCCL, AMD's tuned all-reduce algorithms
  for XGMI-connected MI300/MI355 topologies.
- `NCCL_SOCKET_IFNAME=lo` -- bootstrap-only over loopback; real data plane
  is XGMI.
- The script's own comment (line ~45) calls this out:
  *"Single-node interconnect = AMD Infinity Fabric (xGMI) between GPUs."*

### In Megatron-LM itself -- no direct XGMI references
`grep -ri "xgmi\|infinity.?fabric" Megatron-LM/` returns 0 hits. Megatron-LM
never talks to XGMI directly. Instead:
1. It calls PyTorch `torch.distributed` collectives (AllReduce, AllGather,
   ReduceScatter) for tensor/data/pipeline parallelism.
2. PyTorch routes those through the `nccl` backend, which on ROCm is
   actually **RCCL** (binary-compatible API).
3. RCCL picks the transport per GPU pair: **XGMI peer-to-peer when
   available**, SHM otherwise, network when needed. Since the topology
   shows every pair is 1 XGMI hop, RCCL uses XGMI for every collective on
   this node.

### How much is XGMI actually exercised by the current run?
The current config is `--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1`
(pure DP). With TP=PP=1, the only things hitting XGMI during training are:
1. The DP gradient AllReduce at step boundaries.
2. The distributed-optimizer AllGather of sharded Adam state.

Forward/backward GEMMs stay local per GPU -- no cross-GPU traffic in the
hot path. To stress XGMI, raise TP so attention/MLP matmuls AllReduce on
every layer.

### Why do the env var names start with NCCL_ if we're using RCCL?

RCCL is a deliberate drop-in replacement for NCCL, including keeping the
same environment variable names.

- **API/ABI compatibility by design.** RCCL is a 1:1 port of NCCL for AMD
  GPUs -- every function (`ncclAllReduce`, `ncclSend`, etc.), error code,
  and header is named identically so frameworks like PyTorch and Megatron-LM
  work on AMD without changing any collective call sites.
- **Env vars are part of that contract.** If RCCL renamed `NCCL_P2P_DISABLE`
  to `RCCL_P2P_DISABLE`, every existing script, container, and tuning guide
  written for NVIDIA would silently fail to apply.
- **PyTorch reinforces this.** `torch.distributed.init_process_group(backend="nccl")`
  works unchanged on ROCm -- PyTorch just links against `librccl.so` instead
  of `libnccl.so`.
- **`RCCL_*` is reserved for AMD-only extensions.** Variables with no NCCL
  equivalent get the `RCCL_` prefix.

The split in `work/run.sh`:
- `NCCL_IB_DISABLE`, `NCCL_P2P_DISABLE`, `NCCL_SHM_DISABLE`, `NCCL_PROTO`,
  `NCCL_ALGO`, `NCCL_DEBUG`, `NCCL_SOCKET_IFNAME` -- inherited from NCCL.
- `RCCL_MSCCL_ENABLE` -- AMD-only (MSCCL is AMD's tuned collective scheduler),
  so it gets the `RCCL_` prefix.

---

## What else is in each Megatron-LM log (besides TFLOP/s and comm timers)

Inventory of everything the per-N log file emits, grouped by category. Useful
when reading `logs/sweep_<STAMP>/bench_bf16_n<N>.log`.

### 1. Per-iteration training scalars (the main iter line)

Every `--log-interval` steps, in addition to TFLOP/s/GPU:

- `elapsed time per iteration (ms)` -- raw wall-clock per step (3,467 ms
  steady-state on the 8-GPU run).
- `consumed samples` -- running total.
- `mem usages` -- normalized HBM fraction (~0.636 on 8-GPU).
- `learning rate` -- cosine schedule progression.
- `global batch size` -- sanity check during the sweep.
- `lm loss` -- CE loss on mock data; here it falls 12.05 -> 8.12 over the 50
  iters then plateaus (expected on `--mock-data`).
- `loss scale` -- only meaningful for FP16 (constant 1.0 in BF16).
- `grad norm` -- pre-clip L2 of all gradients (37.6 at iter 1 -> 0.19 at
  iter 50; spikes warn of instability).
- `number of skipped iterations` / `number of nan iterations` --
  non-finite-grad guard.

### 2. Compute timers (per stage, per rank)

Same `--timing-log-level 2` block, non-comm side:

- `forward-compute` / `backward-compute` / `forward-backward` -- local-compute
  fraction of the step (~3,140 ms of 3,471 ms on 8-GPU).
- `batch-generator` -- data-loader time per step (~1.25 ms with `--mock-data`).
- `optimizer-copy-to-main-grad` (bf16->fp32 grad cast),
  `optimizer-inner-step` (Adam math itself),
  `optimizer-copy-main-to-model-params` (fp32->bf16 weight write-back), and
  the rolled-up `optimizer` total (~98.5 ms on 8-GPU).

### 3. Model + memory footprints (printed once, before training)

- `Total number of parameters in billions: 16.22` -- split into transformer
  (15.60 B) and embeddings (0.62 B), plus "most loaded shard" (matters under
  TP/PP).
- `Activation memory footprint per transformer layer (precise, without SP): 1632.0 MB`.
- `Theoretical memory footprints: weight and optimizer=116036.06 MB,
  activation=69522.67 MB, total=185558.73 MB`.
- `[Rank 0] (after N iterations) memory (MB) | allocated | max allocated |
  reserved | max reserved` -- actual HBM after iter 1 and iter 5 (allocated
  116 GB; max-allocated 162 -> 178 GB once the bwd pass holds activations;
  reserved 170 -> 179 GB).

### 4. Startup phase timers (one-shot, before iter 1)

Cold-start cost -- useful for short benchmarks where init dominates:

- `startup-program-entry-spread`, `startup-library-setup`,
  `startup-program-setup`, `startup-in-process-setup`,
  `startup-in-job-setup`.
- `startup-initialize-megatron`, `startup-megatron-init-local`,
  `startup-megatron-init-global`, `startup-set-jit-fusion-options`.
- `all-reduce-start-timestamps-tensor` -- RCCL world handshake.
- `model-and-optimizer-setup` and `train/valid/test-data-iterators-setup` --
  model build + dataset wiring.

### 5. End-of-run evaluation

Even with `--eval-interval 1000000`, the final test/validation eval still
runs at job exit. Each eval prints `Evaluating iter K/100` over 1600 mock
samples and reports a separate validation/test loss. Mostly noise on
`--mock-data`, but the iter-rate during eval is a clean inference-throughput
number.

### 6. Full argument dump

The ~800-line argparse echo near the top of each log records every Megatron
knob (including ones the script didn't set). Authoritative reproducibility
record per N -- including `data_parallel_size`, parallel-group geometry
(`initialized tensor model parallel with size 1` / `pipeline ... size 1`),
and the full `AdamOptimizerConfig` and `DistributedDataParallelConfig`.

### 7. Environment + warnings (worth scanning per run)

- Python / PyTorch / `torch.cuda.device_count()` banner (the in-container check).
- `HIP_VISIBLE_DEVICES` line (added by the sweep `run.sh`) -- confirms each
  sweep point saw the right N devices.
- RCCL `NCCL_DEBUG=WARN` lines: `NUMA auto balancing enabled` (variance
  risk), `Missing "iommu=pt"` (perf/stability), TransformerEngine
  flash-attn-version range warning, Apex falling back to native RoPE. These
  don't change between sweep points but the NUMA one in particular can
  cause rank-time spread to grow at higher N.

### What is **not** in the logs (and you'd need extra wiring for)

- GPU power / energy -- the run shows `log_energy = False`; flipping
  `--log-energy` would add a per-step energy line (pynvml-based).
- HBM bandwidth, kernel-level latency, achieved-vs-theoretical FLOPS per
  kernel -- needs `rocprof` / `omniperf` outside Megatron.
- Tokens/sec -- not printed directly; derive it from
  `global batch size * seq-length / elapsed_time_per_iter` (~18.9k tok/s for
  the 8-GPU run).
- Per-collective bytes / busBW -- would need `NCCL_DEBUG=INFO` and
  `NCCL_DEBUG_SUBSYS=COLL,P2P` (current setting is `WARN`, intentionally
  quieter).
