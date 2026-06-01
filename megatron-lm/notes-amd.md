# Notes — Why AMD let the non-power-of-2 RCCL cliff ship

Companion note to [`summary-rccl.md`](summary-rccl.md). The cliff at N ∈ {5, 6, 7} is a real, measured gap (see summary-rccl §1, §7). This file captures the *why-did-they-let-this-happen* reasoning, which is informed speculation about AMD's priorities — not insider knowledge.

## 1. Their dominant customer pattern doesn't hit it

Who buys MI300X / MI325X / MI355X today?

- **Large AI labs** training frontier LLMs — run at *whole node or whole cluster*, where N is always a multiple of 8.
- **HPC centers** (e.g., El Capitan / Frontier-class deployments) — also run at whole-node multiples.
- **Hyperscalers** (Meta, Microsoft, Oracle) — build their own collective libraries on top of RCCL, often patching tunings themselves.

For these customers, N=8 *is* the case, and inter-node scaling at 8, 16, 32, 64, … burns 99 % of GPU-hours. The shared-node case where one user gets 5 GPUs and another gets 3 — exactly the pattern that hits N=5/6/7 — is a smaller-team / academic / mixed-tenant pattern. AMD optimized where the customer revenue is.

## 2. Tuning is expensive engineering

A tuned MSCCL plan isn't an afternoon's XML edit. It is:

- Pick an algorithm decomposition for each `(topology, N, message size bucket)` cell.
- Measure on real hardware.
- Validate correctness across data types and ops.
- Regression-test across ROCm releases.
- Cover the matrix: `{all_reduce, all_gather, reduce_scatter, broadcast, …} × {N=2,3,4,5,6,7} × {small/med/large size buckets}` — dozens of plan files per topology generation.

RCCL has a smaller engineering team than NCCL at NVIDIA. Under resource constraints, you ship the most-used cell first (`8n` full-node) and backlog the rest. The empirical evidence is right there in `/opt/rocm/share/rccl/msccl-algorithms/`: only `-8n-` files exist.

## 3. The hardware is brand new

MI355X (gfx950) launched October 2025. The image used here (ROCm 6.4.3) is from early 2026. New AMD silicon typically gets full RCCL tuning coverage 1–2 ROCm releases after launch. We are in the "make it work" phase, not the "make it optimal at every N" phase.

## 4. NVIDIA had the same problem — and the fix wasn't tuning, it was switched silicon

DGX-1 P100 (NVLink mesh, no switch) showed similar non-power-of-2 cliffs in the NCCL era. NVIDIA didn't really *fix* it in NCCL; they made it irrelevant by shipping NVSwitch in DGX-2 / A100-SXM / H100. AMD is following the same path with **UALink** in the MI400 generation. From AMD's perspective, the structural fix is hardware (UALink), not a multi-year MSCCL-plan-authoring effort that gets obsoleted the moment a switched fabric ships.

## 5. From AMD's PnL it isn't a ship-stopper

The benchmarks customers compare on are MLPerf, frontier LLM throughput at scale, MFU at full node — *all power-of-2*. No public benchmark scores AMD GPUs at N=5. So this gap doesn't show up on any board deck, doesn't lose a customer at procurement, and doesn't appear in a competitive comparison. It only hurts users *after* they've committed to the platform and try to share a node. That's a poor user-experience outcome but a rational ROI decision for AMD's release planning.

## Is "limits the usage of GPUs a lot" fair?

Partly. If you run full-node training the cliff never bites and the limitation is invisible. If you run mixed-tenant or experiment-mode at an arbitrary GPU count, it really does cap throughput at ~65 % efficiency — exactly the gap measured in summary-1/2 §3. So the impact is real for some workloads, immaterial for others.

## What changes the picture

- **Short-term — you can fix it yourself.** Authoring MSCCL plans for `5n` / `6n` / `7n` and dropping them into `/opt/rocm/share/rccl/msccl-algorithms/` is a few weeks of work for someone who knows the format. The infrastructure is open. AMD welcomes upstream PRs.
- **Medium-term — AMD will likely close it.** As MI355X deployments mature and more shared-node use cases are reported (this kind of bug report is exactly how upstream tuning prioritization happens), expect non-`8n` plans to ship in a future ROCm release.
- **Long-term — UALink eliminates the question.** Once a switched fabric arrives in the MI400 generation, the graph-decomposition problem stops existing and arbitrary N becomes smooth.

## Candid framing

AMD didn't "let" the cliff happen so much as they **prioritized the customer cases that drive their business and bet on the hardware fix (UALink) over backfilling tuning for a generation that will turn over.** It is a defensible call from AMD's side; it is also a real limitation that deserves an upstream bug report. Filing one with the `rccl-tests` numbers from [`summary-rccl.md`](summary-rccl.md) as repro is probably the highest-leverage thing a user can do — it puts the issue on AMD's prioritization radar with data attached.
