# Kimi-K3 — repeatability of the MAD-vs-original gap

`kimi-k3-comparison.md` reported the MAD recipe at **0.91× the original** from a
single run each, and flagged that one run per config cannot support a claim that
size. This run repeats c=64 on both configs to separate signal from noise.

Source: `logs/atom/kimi_repeats_20260820_161638/`

## Per-repeat measurements

| Config | Rep | tok/s | TTFT med (ms) | TPOT med (ms) | completed |
|---|---:|---:|---:|---:|---:|
| `A_original` | 1 | 1,347.2 | 249.9 | 46.14 | 640 |
| `A_original` | 2 | 1,375.4 | 248.3 | 45.79 | 640 |
| `A_original` | 3 | 1,372.8 | 247.9 | 45.84 | 640 |
| `B_mad` | 1 | 1,148.2 | 277.8 | 52.93 | 640 |
| `B_mad` | 2 | 1,193.2 | 280.0 | 52.90 | 640 |
| `B_mad` | 3 | 1,197.5 | 278.9 | 52.62 | 640 |

## Statistics

| Config | n | mean tok/s | stdev | spread (max−min) | rel. spread |
|---|---:|---:|---:|---:|---:|
| `A_original` | 3 | **1,365.2** | 15.6 | 28.2 | 2.1% |
| `B_mad` | 3 | **1,179.7** | 27.3 | 49.3 | 4.2% |

## Verdict

- Original (`A_original`): **1,365.2 ± 15.6** tok/s (n=3)
- MAD (`B_mad`): **1,179.7 ± 27.3** tok/s (n=3)
- **Ratio MAD/original = 0.864×** (single-run estimate was 0.908×)

**The gap is real.** The 185.5 tok/s difference between configs is larger than twice the worst within-config spread (49.3 tok/s), so it is not explained by run-to-run variation at this sample size. The single-run 0.91× estimate holds up.

## Caveats

- **Repeats share a server process.** The model is loaded once per config and the benchmark run N times against the same live server. This measures benchmark-to-benchmark variance, **not** full cold-start variance — real deployment variance (load placement, memory layout, JIT state) could be larger.
- **Single concurrency (c=64).** The gap could differ at other batch sizes; this tests only the point the original comparison used.
- **Small n.** Three repeats bounds gross noise, not subtle systematic effects.

## Source data

| Per-repeat JSON / logs | `logs/atom/kimi_repeats_20260820_161638/<config>_rep<N>.{json,log}` |
|---|---|
| Server logs | `logs/atom/kimi_repeats_20260820_161638/<config>_server.log` |
| This table as CSV | `results/kimi-k3-repeats.csv` |
| Single-run comparison this tests | `kimi-k3-comparison.md` |

