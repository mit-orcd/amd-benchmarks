# fp4 N>=5 scaling investigation

Source: `/home/amd/shaohao/amd-benchmarks/amd-cloud/logs/rvs/fp4_investigation_20260813_221751`

Per repeat: per-GPU peak TFLOPS (RVS gpu-id space) alongside the concurrently sampled clock/power distribution (rocm-smi index space -- these two id spaces are NOT the same GPU numbering and are not directly joined here; the clock/power columns are a same-run sanity check, not a per-GPU-id correlation).

| N | repeat | TFLOPS spread (min-max, GPU-id space) | sclk spread MHz (rocm-smi index space) | power spread W (rocm-smi index space) |
|---|---:|---|---|---|
| 5 | 1 | 3543-3916 (spread 10%) | 119-1875 | 245-767 |
| 5 | 2 | 2345-3618 (spread 35%) | 114-1934 | 245-741 |
| 5 | 3 | 2013-3252 (spread 38%) | 111-1923 | 245-749 |
| 6 | 1 | 1394-3442 (spread 60%) | 117-1932 | 246-663 |
| 6 | 2 | 1447-3547 (spread 59%) | 114-1892 | 246-724 |
| 6 | 3 | 1611-3336 (spread 52%) | 111-1890 | 245-663 |
| 7 | 1 | 2148-2524 (spread 15%) | 114-1851 | 257-595 |
| 7 | 2 | 1380-2913 (spread 53%) | 116-1892 | 257-647 |
| 7 | 3 | 1419-3913 (spread 64%) | 123-1888 | 258-657 |
| 8 | 1 | 1775-3302 (spread 46%) | 1799-1899 | 436-542 |
| 8 | 2 | 1782-3671 (spread 51%) | 1761-1874 | 418-653 |
| 8 | 3 | 1812-3362 (spread 46%) | 1774-1843 | 421-609 |

## Consistency across repeats: is it the same GPU every time?

For each N, which gpu-id landed in the bottom half of per-GPU TFLOPS, per repeat. If the same id(s) appear across all repeats at a given N, that is a deterministic, likely hardware- or topology-correlated effect. If the low performer changes between repeats, that points at non-determinism in the launch/sync path instead.

### N=5
- repeat 1: bottom half = {17010, 42583}
- repeat 2: bottom half = {1590, 36479}
- repeat 3: bottom half = {17010, 42583}
- gpu-ids in the bottom half in EVERY repeat: {none} (0% overlap) -> **inconsistent — likely non-deterministic/software-correlated**

### N=6
- repeat 1: bottom half = {1590, 36479, 42583}
- repeat 2: bottom half = {1590, 17010, 27226}
- repeat 3: bottom half = {17010, 27226, 36479}
- gpu-ids in the bottom half in EVERY repeat: {none} (0% overlap) -> **inconsistent — likely non-deterministic/software-correlated**

### N=7
- repeat 1: bottom half = {1590, 36479, 51771}
- repeat 2: bottom half = {17010, 27226, 51771}
- repeat 3: bottom half = {17010, 36479, 51771}
- gpu-ids in the bottom half in EVERY repeat: {51771} (20% overlap) -> **inconsistent — likely non-deterministic/software-correlated**

### N=8
- repeat 1: bottom half = {11806, 1590, 17010, 36479}
- repeat 2: bottom half = {11806, 17010, 27226, 51771}
- repeat 3: bottom half = {1590, 17010, 36479, 57875}
- gpu-ids in the bottom half in EVERY repeat: {17010} (14% overlap) -> **inconsistent — likely non-deterministic/software-correlated**

## How to read this

- **High consistency + low performers also show depressed clocks/power**: a real per-die thermal or power effect (e.g. VRM zone, cooling asymmetry). Not a bug.
- **High consistency + clocks look normal across the board**: a deterministic effect not explained by clocks — worth checking topology (NUMA/XGMI placement of those specific dies) rather than power.
- **Low consistency (different GPU low each repeat)**: points at non-determinism in RVS's parallel gst launch or in the fp4 MXFP4 kernel path under concurrent multi-GPU load — a software/scheduling issue, not a hardware one.

## Caveat

rocm-smi's GPU index and RVS's internal gpu id are different numbering schemes and are not joined in this analysis (see the docstring). A future pass could resolve this via `rvs -g` output order, at which point the clock/power columns could be attributed to specific gpu ids rather than reported as a same-run range.

