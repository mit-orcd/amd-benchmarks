# Kimi-K3 — profiler trace summary (next-step #2)

Replaces the residual-based estimate in `kimi-k3-improve.md` §3 with measured
kernel time. That section attributed ~80% of step time to "prefill + scheduling"
by subtracting estimated costs — this is the direct measurement instead.

Traces: `/mnt/scratch/shaohao/traces/kimi_20260820_041644` (8 file(s), 95 MB) — kept off-repo,
driver log `kimi_profile_20260820_041644`.

## No GPU kernel events found

Parsed 702,510 trace events from `model_ts_20260820_042332_595.pt.trace.json.gz` but none carried a GPU kernel category. The trace may be CPU-only, or the categories differ in this kineto version. Raw traces retained; try ATOM's own `tools/analyze_trace_summary.py`.

