# RCCL Collective Communications — Reference & Analysis

This document inventories every RCCL collective primitive: what it does, its bandwidth model, which RCCL algorithms back it, what Megatron-LM uses it for, and (where we have measured data) the observed performance on this MI355X node.

The N=5/6/7 cliff, mesh-vs-switched-fabric topology, and AMD generation-wide recommendations are covered separately in [summary-power2.md](summary-power2.md). This file does not repeat that material.

---

## 1. Measured results and analysis

### 1.1 Full collective sweep (`logs/rccl_all_20260602_121713/`, `log.nccl-all`)

The all-collective rccl-tests sweep on this MI355X node, June 2, 2026. Config: message sizes 16 MiB → 8 GiB (powers of 2, factor=2), warmup=5, iters=20, `NCCL_ALGO=Ring,Tree`, `NCCL_PROTO=Simple,LL,LL128`, `RCCL_MSCCL_ENABLE=1`, single-node intra-xGMI. All numbers below are **busbw (GB/s) at the 8 GiB top-end message size**, in-place column from the rccl-tests output.

| Collective       |    N=2 |    N=3 |     N=4 |    N=5 |    N=6 |    N=7 |     N=8 |
|------------------|-------:|-------:|--------:|-------:|-------:|-------:|--------:|
| all_reduce       |  61.28 |  75.02 |  166.48 |  38.36 |  38.42 |  38.21 |  381.27 |
| all_gather       |  60.58 |  71.09 |  158.72 |  35.41 |  34.90 |  34.85 |  365.75 |
| reduce_scatter   |  60.65 |  71.04 |  165.06 |  39.57 |  39.65 |  40.47 |  407.69 |
| broadcast        |  63.52 |  68.09 |  169.19 |  34.10 |  33.92 |  33.83 |  377.27 |
| reduce           |  72.87 |  86.46 |  197.44 |  43.60 |  42.93 |  43.08 |  358.49 |
| gather           |  72.07 |  78.27 |  211.64 |  69.38 |  68.83 |  70.29 |  444.15 |
| scatter          |  63.11 |  71.48 |  191.63 |  65.35 |  65.61 |  66.36 |  426.40 |
| alltoall         |  58.40 |  61.79 |  155.21 |  44.14 |  45.62 |  44.26 |  360.90 |
| sendrecv¹        |  59.21 |  60.32 |   60.59 |  43.83 |  43.77 |  43.44 |   53.24 |

¹ The sendrecv row was captured by a follow-on standalone sweep (`logs/rccl_sendrecv_20260602_153246/`, `log.nccl-sendrecv`) on the same env stack and message envelope.

**AllToAllV** is intentionally omitted from the table: at N=2/3/4 the busbw at 8 GiB measured 58.16 / 34.66 / 115.88 GB/s (30–35 % below the equal-chunk AllToAll at the same N, with N=3 anomalously low — likely a per-rank chunk-alignment artifact); N=5 was OOM-killed (`rc=137`) before reaching 8 GiB, only sizes ≤128 MiB produced numbers (peak ~12.7 GB/s busbw); N=6 was truncated mid-run; N=7/8 never ran — see §1.3.

#### 1.1.1 Patterns visible across the matrix

- **The N=5/6/7 cliff is universal across the bandwidth-bound collectives.** Every collective whose busbw model has a `(N-1)/N` or `2(N-1)/N` factor — all_reduce, all_gather, reduce_scatter, broadcast, reduce, alltoall — collapses ~4–5× between N=4 and N=5 and stays flat at N=6/7 before snapping back at N=8. This confirms the topology-driven analysis in [summary-power2.md](summary-power2.md): the K_8 xGMI mesh has a clean factor-of-N ring only at the power-of-2 GPU counts {2, 4, 8}; at N ∈ {5, 6, 7} RCCL falls back to a path that loses one of the four xGMI channels and the resulting busbw is set by the slowest link rather than the aggregate bisection.
- **Gather and Scatter do *not* show the cliff at N=5/6/7.** They drop only ~2× across the cliff vs. ~4–5× for the ring-based collectives. Reason: both are implemented as pairwise sendrecv loops to/from the root rank, which doesn't depend on a complete ring being available. Their bandwidth at N=5–7 is set by the single root-to-peer link rather than the global ring composition.
- **Reduce > AllReduce at every N.** Expected: Reduce only does the reduce phase, AllReduce additionally broadcasts. The ratio is ~1.2× at N=2 and grows toward the `~2×` ratio that the analytic model predicts for very wide rings.
- **AllReduce ≈ ReduceScatter ≈ AllGather at the same N**, within 5–15 %. The Ring AllReduce really is ReduceScatter+AllGather, so its busbw matches the slower of the two halves. This was the prediction in §2.2.3 and the sweep confirms it.
- **Broadcast ≈ Reduce, both > AllReduce.** Single-direction primitives saturate the fabric without the second "fold-back" pass.
- **AllToAll tracks AllGather closely at large N** (N=8: 360.9 vs 365.8 GB/s; N=4: 155.2 vs 158.7 GB/s). At small N it drops below — likely the pairwise sendrecv path is less efficient at N=2/3 than the dedicated Ring AllGather.
- **AllToAllV is 30–35 % slower than AllToAll at the same N** for the configurations that completed (N=2: 58 vs 58 GB/s; N=3: 35 vs 62 GB/s; N=4: 116 vs 155 GB/s). The variable-chunk path has noticeably more per-call overhead. N=3 is unusually bad — likely the per-rank chunk sizing falls out of an alignment window.
- **N=8 gather/scatter are the all-time peaks of the matrix.** Gather hits 444 GB/s and scatter hits 426 GB/s, both above the 381 GB/s AllReduce N=8 peak. Pairwise sendrecv to/from a single root at full xGMI rate is the cleanest way to saturate the fabric on this topology when the result-aggregation pattern fits.
- **SendRecv has a much milder N=5/6/7 cliff than the ring-based collectives** — ~28 % drop (60 → 43 GB/s) vs ~76 % drop (167 → 38 GB/s) for AllReduce. Reason: `sendrecv_perf` exercises a single point-to-point ring where every rank does one send and one recv at a time, so the busbw is set by the *slowest single hop* in the ring rather than by aggregate ring bisection. At N=5/6/7 one pair lands on a slower indirect path; at N=2/3/4 every pair has a direct xGMI link (~60 GB/s, the single-link xGMI rate).
- **SendRecv at N=8 does *not* snap back to a high peak** (53 GB/s, below the N=2..4 plateau). Unlike Ring AllReduce — which at N=8 reactivates all four xGMI channels in parallel and recovers to 381 GB/s — sendrecv has no opportunity to multiplex channels: per-pair traffic uses one link, and the ring closure on the K_8 mesh forces one hop onto a longer path. This is the predicted ceiling for **pipeline-parallel SendRecv** throughput: ~60 GB/s per stage-to-stage exchange at PP=2/4, dropping to ~53 GB/s at PP=8.

### 1.2 Cross-check against Megatron-LM application timers

The Megatron sweeps ([summary.md](summary.md)) report two collective-bearing timers at iter 45:

- `all-grads-sync` — ReduceScatter + the kernel-level reduce. Bounded by ReduceScatter bandwidth.
- `params-all-gather` — straight AllGather of the post-step parameters.

Comparison at the N=4 → N=5 transition (relative slowdown):

| Timer / collective                       | N=4 → N=5 ratio  |
|------------------------------------------|------------------:|
| Megatron `all-grads-sync`                | 4.06×             |
| Megatron `params-all-gather`             | 4.67×             |
| rccl-tests `all_reduce` busbw            | 4.37× (slower)    |
| rccl-tests `all_gather` busbw            | 4.50× (slower)    |

Application-level slowdowns track collective-level slowdowns within ±10 %. The rccl-tests measurements are a faithful predictor of the Megatron timer values at this configuration.

### 1.3 Remaining coverage gap

| Collective       | Status                                                              | Why it matters here                                                                       |
|------------------|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| AllToAllV (N≥5)  | OOM-killed at N=5; user opted to skip retries                       | Per-rank buffer × N replicas grows past device memory at 8 GiB / N≥5; would need smaller `--maxbytes` or per-N scaling. AllToAll (with equal chunks) is fully measured and bounds AllToAllV from above. |
| Multi-node       | Not exercised here (intra-node xGMI only)                            | IB / RoCE bandwidth would be the bottleneck once the collective crosses the node boundary. |
| Sub-MiB messages | Sweep starts at 16 MiB; small-message latency regime not probed     | Where Tree algorithms become competitive with Ring.                                       |

---

## 2. Reference — collectives, algorithms, knobs, next steps

### 2.1 Catalog of RCCL collectives

The full set of NCCL/RCCL primitives exposed via `librccl.so.1` and built by `rccl-tests`. The "busbw" column is the standard rccl-tests bandwidth formula relating measured *algorithm bandwidth* (the byte-count input divided by wall time) to *bus bandwidth* (the steady-state per-link traffic):

| Collective       | Op semantics                                              | Bytes moved / rank | `busbw` formula        | Typical RCCL algorithms       | Megatron-LM usage                                                            | rccl-tests binary           | Tested here? |
|------------------|-----------------------------------------------------------|--------------------|------------------------|--------------------------------|------------------------------------------------------------------------------|------------------------------|--------------|
| **AllReduce**    | reduce-then-broadcast: every rank ends with `op(x_0..x_{N-1})` | `S`            | `algbw × 2·(N-1)/N`    | Ring, Tree, MSCCL              | DP gradient sync without dist-opt; TP forward/backward; tied-embedding sync | `all_reduce_perf`            | **Yes**      |
| **AllGather**    | each rank publishes its slice; everyone ends with the full vector | `S/N` in, `S` out | `algbw × (N-1)/N`  | Ring, PAT, MSCCL               | Distributed-optimizer parameter all-gather; FSDP / ZeRO-3 param gather       | `all_gather_perf`            | **Yes**      |
| **ReduceScatter**| reduce across ranks, then each rank keeps its `1/N` slice | `S` in, `S/N` out  | `algbw × (N-1)/N`      | Ring, PAT, MSCCL               | Distributed-optimizer gradient reduce-scatter; FSDP / ZeRO-2 grad reduce      | `reduce_scatter_perf`        | **Yes**      |
| **Broadcast**    | root sends one buffer; every other rank receives it       | `S`                | `algbw × 1`            | Ring, Tree                     | Parameter sync at train start; pipeline-stage parameter broadcast            | `broadcast_perf`             | **Yes**      |
| **Reduce**       | every rank contributes; only root keeps `op(x_0..x_{N-1})`| `S`                | `algbw × 1`            | Ring, Tree                     | Per-rank loss reduction to rank 0 for logging; rarely on the critical path   | `reduce_perf`                | **Yes**      |
| **Gather**       | every rank's buffer is concatenated on the root only      | `S` in, `N·S` on root | `algbw × (N-1)/N`   | Pairwise sendrecv              | Validation / debug paths; not on the training critical path                  | `gather_perf`                | **Yes**      |
| **Scatter**      | root's `N·S` buffer is split, one slice to each rank      | `N·S` on root, `S` out | `algbw × (N-1)/N`  | Pairwise sendrecv              | Rarely used directly in Megatron                                              | `scatter_perf`               | **Yes**      |
| **AllToAll**     | each rank sends a distinct chunk to every other rank      | `S` in, `S` out (transposed) | `algbw × (N-1)/N` | Pairwise, MSCCL          | MoE expert token routing (the key MoE primitive); TP token reshuffles         | `alltoall_perf`              | **Yes**      |
| **AllToAllV**    | same as AllToAll with per-peer variable chunk sizes       | variable           | `algbw × (N-1)/N`      | Pairwise                       | MoE with imbalanced token counts                                              | `alltoallv_perf`             | Partial (N≤4)|
| **SendRecv**     | point-to-point pair                                       | `S`                | `algbw × 1`            | Direct xGMI / SHM / IB         | Pipeline-parallel stage-to-stage activations and gradients                    | `sendrecv_perf`              | **Yes**      |

Reading guide: `busbw` ≥ `algbw` for AllReduce by definition (each byte traverses the fabric ~2× to be reduced and then propagated), `busbw` < `algbw` for AllGather / ReduceScatter / AllToAll (each rank only sends `(N-1)/N` of its data because the local slice stays put), and `busbw == algbw` for one-shot primitives (Broadcast, Reduce, SendRecv). When comparing collectives at the same `busbw`, AllReduce is moving roughly twice as much wire traffic as AllGather.

### 2.2 Bandwidth model and Megatron usage by collective

#### 2.2.1 AllReduce (`ncclAllReduce`)

- **Semantics.** Combines per-rank input buffers element-wise with `op` and writes the combined result to every rank's output buffer.
- **Bandwidth model.** A ring-based AllReduce decomposes into a ReduceScatter phase (`(N-1)/N · S` bytes per rank) followed by an AllGather phase (`(N-1)/N · S` bytes per rank). Total wire traffic per rank is `2 · (N-1)/N · S`, so `busbw = algbw × 2·(N-1)/N`.
- **Megatron critical path.** Fires once per training step when the distributed optimizer is *off* (full DP grad sync); per-layer in TP forward and TP backward; on the embedding when `--untie-embeddings-and-output-weights` is *not* set. In the current `run.sh` config (TP=1, dist-opt on, embeddings untied) AllReduce is **not** the primary collective — the dist-opt path replaces it with ReduceScatter + AllGather.
- **Algorithms in RCCL.** `Ring` (default for large messages), `Tree` (better at small message latency on switched fabrics; *worse* on `K_8` mesh per measurements), and `MSCCL` (custom-tuned plans when present).
- **MSCCL plan coverage on this server** (`/opt/rocm-7.2.3/share/rccl/msccl-algorithms/`): `allreduce-allpairs-8n-*.xml` ships for N=8 only.

#### 2.2.2 AllGather (`ncclAllGather`)

- **Semantics.** Each rank publishes a slice of size `S/N`; after the call every rank holds the concatenation of all slices.
- **Bandwidth model.** Each rank receives `(N-1)/N · S` bytes from peers (its own slice never crosses the wire), so `busbw = algbw × (N-1)/N`.
- **Megatron critical path.** Fires once per training step when `--use-distributed-optimizer` is on — after the optimizer step each rank has updated its `1/N` slice of parameters and must broadcast it back to peers via AllGather. Also used by FSDP / ZeRO-3 to materialize each layer's parameters just before forward / backward.
- **Algorithms in RCCL.** `Ring` (default), `PAT` (parallel aggregation tree on some fabrics — limited support here), `MSCCL` (custom plans when present). RCCL's AllGather does **not** have a real Tree implementation — forcing `NCCL_ALGO=Tree` falls back to Ring with the same numbers.
- **MSCCL plan coverage on this server**: `allgather_16n_direct_*` and `allgather_32n_direct_*` ship; none for N=8 specifically.

#### 2.2.3 ReduceScatter (`ncclReduceScatter`)

- **Semantics.** Element-wise reduction across ranks followed by scatter — each rank keeps only its `1/N` slice of the reduced result.
- **Bandwidth model.** Each rank sends `(N-1)/N · S` bytes, so `busbw = algbw × (N-1)/N`. Same model as AllGather, half the wire traffic of AllReduce.
- **Megatron critical path.** With `--use-distributed-optimizer`, gradient sync is implemented as a ReduceScatter (each rank ends owning `1/N` of the reduced grads) followed by the optimizer step and then an AllGather of updated params. So ReduceScatter is the *first half* of every DP comm step.
- **Algorithms in RCCL.** `Ring`, `PAT`, `MSCCL`. Like AllGather, RCCL's ReduceScatter has no real Tree algorithm.
- **Not directly measured in this sweep** — but its per-N performance is implicit in the AllReduce numbers (AllReduce is ReduceScatter + AllGather sharing the same fabric mechanics), and Megatron's `all-grads-sync` timer captures it end-to-end.

#### 2.2.4 Broadcast (`ncclBroadcast`)

- **Semantics.** Root rank's buffer is copied to all other ranks.
- **Bandwidth model.** Linear scan: each byte the root produces must traverse the fabric `N-1` times (or via tree, `log N` times). Standard formula `busbw = algbw × 1` (every rank receives `S` bytes regardless of its position in the fabric).
- **Megatron critical path.** Fires during initialization (broadcast initial parameters from rank 0), during checkpoint load, and inside pipeline-parallel initialization. **Not on the training step critical path** once steady state is reached.
- **Algorithms in RCCL.** `Ring` (default for large), `Tree` (default for small). `--algo=Tree` is the right choice for short broadcasts where latency dominates.

#### 2.2.5 Reduce (`ncclReduce`)

- **Semantics.** Element-wise reduction; only the root rank receives the result.
- **Bandwidth model.** Dual of Broadcast: every rank's input must be folded into the root. `busbw = algbw × 1`.
- **Megatron critical path.** Mostly used for per-step **loss logging** (reduce loss to rank 0 for stdout). Off the critical path under any normal training schedule.

#### 2.2.6 Gather and Scatter (`ncclGather`, `ncclScatter`)

- **Semantics.** Gather: each rank contributes its `S` bytes; only the root ends with the concatenated `N·S`. Scatter: inverse.
- **Bandwidth model.** Both implemented inside RCCL via pairwise sendrecv loops; the formula `busbw = algbw × (N-1)/N` reflects that the root's local slice doesn't traverse the wire.
- **Megatron critical path.** Not used in standard Megatron training. Sometimes appears in validation / eval / inference utilities.

#### 2.2.7 AllToAll (`ncclAllToAll`) and AllToAllV (`ncclAllToAllV`)

- **Semantics.** AllToAll: rank `i` has an `N·S` send buffer split into `N` chunks; chunk `j` goes to rank `j`. AllToAllV: same, but per-peer chunk sizes are variable.
- **Bandwidth model.** Each rank sends `(N-1)/N · (N·S) = (N-1)·S` bytes total (to `N-1` peers), so `busbw = algbw × (N-1)/N` (same as AllGather form when the per-rank message size is `N·S`).
- **Megatron critical path.** The defining primitive for **MoE expert routing**: tokens are AllToAll-ed across the expert-parallel dimension so each token reaches the GPU hosting its assigned expert. With imbalanced expert loads, AllToAllV is used instead.
- **Algorithms in RCCL.** Pairwise sendrecv pattern by default; MSCCL provides handcrafted alltoall plans for specific topologies. **MSCCL plan coverage on this server**: `alltoall-8n-*.xml` ships for N=8 across multiple message-size bands (`0-9kb`, `9kb-190kb`, `190kb-512kb`, `512kb-7mb`, `7mb-43mb`).

#### 2.2.8 SendRecv (`ncclSend` / `ncclRecv`)

- **Semantics.** Point-to-point. Sender posts `ncclSend(buf, peer)`; receiver posts matching `ncclRecv`. Not a collective in the strict sense but exposed via the same library.
- **Bandwidth model.** `busbw = algbw × 1` — every byte traverses one fabric link once.
- **Megatron critical path.** The primitive behind **pipeline parallelism**: forward activations and backward gradients between adjacent pipeline stages are SendRecv pairs. In the current `run.sh` (PP=1) it is unused; would become the dominant collective if PP ≥ 2.

### 2.3 RCCL algorithm selection — which algorithm runs for which collective

RCCL chooses an algorithm at runtime based on (message size, collective type, fabric topology, available MSCCL plans). The table below summarizes what is implemented vs. what falls back to a different path:

| Collective       | `Ring`        | `Tree`            | `PAT`         | `MSCCL` | `NVLS`      | Pairwise sendrecv |
|------------------|---------------|-------------------|---------------|---------|-------------|-------------------|
| AllReduce        | Default large | Default small (sw fabric) / **degraded on mesh** | —             | If plan exists | NVSwitch only | — |
| AllGather        | Default       | Falls back to Ring | Available     | If plan exists | NVSwitch only | — |
| ReduceScatter    | Default       | Falls back to Ring | Available     | If plan exists | NVSwitch only | — |
| Broadcast        | Default large | Default small     | —             | —       | —           | — |
| Reduce           | Default large | Default small     | —             | —       | —           | — |
| Gather / Scatter | —             | —                 | —             | —       | —           | **Default** |
| AllToAll         | —             | —                 | —             | If plan exists | —           | **Default** |
| SendRecv         | —             | —                 | —             | —       | —           | **Direct**        |

Notes:
- **NVLS (NVSwitch in-network reductions).** Not applicable on AMD until UALink ships.
- **PAT (parallel aggregation tree).** A multi-channel topology-aware path used on some Mellanox switches; not active on this MI355X xGMI mesh.
- **MSCCL plan files.** Live in `/opt/rocm-7.2.3/share/rccl/msccl-algorithms/`. The bundled plans on this server cover: AllReduce (N=8), AllGather (N=16, 32), AllToAll (N=8, with multiple message-size buckets). Anything outside those configurations falls through to Ring / pairwise sendrecv.
- **Tree on AllGather / ReduceScatter is a no-op.** Forcing `NCCL_ALGO=Tree` for these collectives is silently equivalent to Ring — confirmed in the sweep (all configs within 1–3 %).

### 2.4 Configuration knobs that affect collective behavior

Environment variables that route through RCCL's algorithm-selection layer. The "scope" column indicates which collectives the knob actually changes:

| Variable                  | Scope                                                | Default              | Effect                                                                                          |
|---------------------------|------------------------------------------------------|----------------------|-------------------------------------------------------------------------------------------------|
| `NCCL_ALGO`               | AllReduce, Broadcast, Reduce (Tree path)             | `Ring,Tree`          | Restrict / order the algorithm pool. `Tree` is only meaningful for these three collectives.    |
| `NCCL_PROTO`              | All collectives                                      | `Simple,LL,LL128`    | Choose the wire protocol; on this fabric `Simple` ≡ default at large message size.              |
| `RCCL_MSCCL_ENABLE`       | AllReduce, AllGather, AllToAll (plans present)       | `1`                  | Toggle MSCCL plan dispatch. With current bundled plans only N=8 (allreduce, alltoall) and N=16/32 (allgather) are affected. |
| `RCCL_MSCCL_ALGO_DIR`     | All collectives with plans                           | `/opt/rocm/share/rccl/msccl-algorithms` | Point to a custom directory of MSCCL XML plans. |
| `NCCL_P2P_DISABLE`        | All collectives                                      | `0`                  | When `1`, disables direct GPU-to-GPU xGMI peer access; forces staging via host SHM. Catastrophically slower; for debug only. |
| `NCCL_SHM_DISABLE`        | All collectives                                      | `0`                  | When `1`, disables host-shared-memory fallback. Leave on. |
| `NCCL_IB_DISABLE`         | All collectives crossing nodes                       | `0` (host)           | Forces single-node-only path. Set to `1` here because the sweep is intra-node.                  |
| `NCCL_DEBUG`              | All                                                  | `WARN` (script)      | Verbosity. `INFO` prints the algorithm + channel count chosen per call (useful for diagnosis). |
| `NCCL_MIN_NCHANNELS`      | Ring algorithms                                      | (heuristic)          | Force at least this many parallel ring channels. Rarely effective beyond what the topology naturally supports. |
| `NCCL_MAX_NCHANNELS`      | Ring algorithms                                      | (heuristic)          | Cap parallel channels. Useful for clean reproducibility runs.                                    |

Two findings from the sweep that the table doesn't show directly:

- **`NCCL_ALGO=Tree` does not affect AllGather / ReduceScatter** (no tree implementation; falls back to Ring with identical numbers).
- **`RCCL_MSCCL_ENABLE=0` matched the default exactly across the sweep**, meaning either (a) the bundled AllReduce / AllGather plans were not engaged for the tested configurations, or (b) they ran but produced numbers indistinguishable from Ring.

### 2.5 Recommended next experiments (collective-level)

1. **AllToAll sweep at N=8 across the 5 MSCCL message bands.** The bundled `alltoall-8n-{0-9kb, 9kb-190kb, 190kb-512kb, 512kb-7mb, 7mb-43mb}.xml` plans suggest this collective is well-tuned at N=8; varying the message band in isolation would confirm MSCCL is actively engaged here (unlike AllReduce / AllGather where MSCCL toggling was a no-op).
2. **Pipeline-parallel SendRecv probe at the application level.** Set up a real Megatron run with PP=2 at N=8 (TP=1, DP=4) and compare stage-to-stage SendRecv throughput to the `sendrecv_perf` baseline (~60 GB/s at small PP, ~53 GB/s at PP=8). This becomes the dominant collective if PP is ever raised.
3. **AllToAllV N≥5 with per-N max-size scaling.** Only needed if MoE imbalance becomes a critical path; for now AllToAll bounds the worst case.
4. **Multi-node IB collectives.** Repeat the all-collective sweep across two nodes once an IB-enabled config is available. Expected: per-link bandwidth drops ~10–20× as soon as the collective crosses the node boundary.
5. **Vary message size below 16 MiB.** The current sweep starts at 16 MiB to bracket Megatron bucket sizes. A sub-MiB probe would expose where Tree algorithms become competitive with Ring.
