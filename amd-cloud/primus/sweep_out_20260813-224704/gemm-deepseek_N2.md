# DeepSeek GEMM Benchmark Report

- Model: Custom
- Date: 2026-08-13 22:59:45
- Duration per shape: 5s

## Configuration
- seqlen: 4096
- hidden_size: 4096
- intermediate_size: 12288
- kv_lora_rank: 512
- moe_intermediate_size: 1536
- num_attention_heads: 64
- num_experts_per_tok: 6
- n_routed_experts: 128
- n_shared_experts: 2
- q_lora_rank: None
- dtype: bf16

## GEMM Shapes (M, N, K)
- attn_q: (4096, 12288, 4096)
- attn_kv_down: (4096, 576, 4096)
- attn_kv_up: (4096, 16384, 512)
- attn_out: (4096, 4096, 8192)
- router: (4096, 128, 4096)
- shared_gateup: (4096, 24576, 4096)
- shared_down: (4096, 4096, 12288)
- moe_gateup: (192, 3072, 4096)
- moe_down: (192, 4096, 1536)
- vocab: (4096, 128256, 4096)

## Phases
- fwd
- wgrad
- dgrad

| host | world | rank | attn_kv_down_dgrad | attn_kv_down_fwd | attn_kv_down_wgrad | attn_kv_up_dgrad | attn_kv_up_fwd | attn_kv_up_wgrad | attn_out_dgrad | attn_out_fwd | attn_out_wgrad | attn_q_dgrad | attn_q_fwd | attn_q_wgrad | moe_down_dgrad | moe_down_fwd | moe_down_wgrad | moe_gateup_dgrad | moe_gateup_fwd | moe_gateup_wgrad | router_dgrad | router_fwd | router_wgrad | shared_down_dgrad | shared_down_fwd | shared_down_wgrad | shared_gateup_dgrad | shared_gateup_fwd | shared_gateup_wgrad | vocab_dgrad | vocab_fwd | vocab_wgrad |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mi355-gpu-33 | 2 | 0 | 849.62TF/s / 1889.89GB/s / 0.022748s / AI=449.56 | 578.89TF/s / 1287.69GB/s / 0.033387s / AI=449.56 | 537.44TF/s / 1195.49GB/s / 0.035962s / AI=449.56 | 915.45TF/s / 2067.37GB/s / 0.075066s / AI=442.81 | 1008.87TF/s / 2278.33GB/s / 0.068115s / AI=442.81 | 1087.48TF/s / 2455.86GB/s / 0.063191s / AI=442.81 | 1476.45TF/s / 901.15GB/s / 0.186175s / AI=1638.40 | 1559.89TF/s / 952.08GB/s / 0.176216s / AI=1638.40 | 1453.71TF/s / 887.27GB/s / 0.189088s / AI=1638.40 | 1563.49TF/s / 890.66GB/s / 0.263716s / AI=1755.43 | 1545.91TF/s / 880.65GB/s / 0.266714s / AI=1755.43 | 1465.62TF/s / 834.91GB/s / 0.281326s / AI=1755.43 | 171.99TF/s / 1049.77GB/s / 0.014046s / AI=163.84 | 197.55TF/s / 1205.73GB/s / 0.012230s / AI=163.84 | 205.16TF/s / 1252.20GB/s / 0.011776s / AI=163.84 | 320.98TF/s / 1854.64GB/s / 0.015053s / AI=173.07 | 266.57TF/s / 1540.26GB/s / 0.018126s / AI=173.07 | 313.70TF/s / 1812.57GB/s / 0.015403s / AI=173.07 | 300.15TF/s / 2491.45GB/s / 0.014310s / AI=120.47 | 257.80TF/s / 2139.98GB/s / 0.016660s / AI=120.47 | 243.78TF/s / 2023.56GB/s / 0.017618s / AI=120.47 | 1499.27TF/s / 854.08GB/s / 0.275011s / AI=1755.43 | 1616.28TF/s / 920.73GB/s / 0.255103s / AI=1755.43 | 1462.46TF/s / 833.11GB/s / 0.281934s / AI=1755.43 | 1545.07TF/s / 817.30GB/s / 0.533720s / AI=1890.46 | 1416.88TF/s / 749.49GB/s / 0.582007s / AI=1890.46 | 1473.01TF/s / 779.18GB/s / 0.559828s / AI=1890.46 | 1559.80TF/s / 773.79GB/s / 2.759036s / AI=2015.81 | 1384.17TF/s / 686.66GB/s / 3.109120s / AI=2015.81 | 1405.93TF/s / 697.45GB/s / 3.060996s / AI=2015.81 |
| mi355-gpu-33 | 2 | 1 | 865.44TF/s / 1925.08GB/s / 0.022332s / AI=449.56 | 591.58TF/s / 1315.90GB/s / 0.032671s / AI=449.56 | 551.23TF/s / 1226.15GB/s / 0.035062s / AI=449.56 | 918.85TF/s / 2075.03GB/s / 0.074789s / AI=442.81 | 985.71TF/s / 2226.03GB/s / 0.069716s / AI=442.81 | 1069.86TF/s / 2416.06GB/s / 0.064232s / AI=442.81 | 1460.41TF/s / 891.36GB/s / 0.188220s / AI=1638.40 | 1537.14TF/s / 938.20GB/s / 0.178824s / AI=1638.40 | 1440.36TF/s / 879.13GB/s / 0.190839s / AI=1638.40 | 1527.52TF/s / 870.17GB/s / 0.269925s / AI=1755.43 | 1522.14TF/s / 867.10GB/s / 0.270880s / AI=1755.43 | 1431.79TF/s / 815.64GB/s / 0.287973s / AI=1755.43 | 168.74TF/s / 1029.89GB/s / 0.014318s / AI=163.84 | 172.01TF/s / 1049.84GB/s / 0.014046s / AI=163.84 | 183.24TF/s / 1118.40GB/s / 0.013185s / AI=163.84 | 308.85TF/s / 1784.53GB/s / 0.015645s / AI=173.07 | 258.17TF/s / 1491.68GB/s / 0.018716s / AI=173.07 | 306.91TF/s / 1773.30GB/s / 0.015744s / AI=173.07 | 307.37TF/s / 2551.38GB/s / 0.013973s / AI=120.47 | 258.75TF/s / 2147.87GB/s / 0.016599s / AI=120.47 | 258.49TF/s / 2145.66GB/s / 0.016616s / AI=120.47 | 1475.90TF/s / 840.76GB/s / 0.279366s / AI=1755.43 | 1588.30TF/s / 904.79GB/s / 0.259597s / AI=1755.43 | 1446.83TF/s / 824.21GB/s / 0.284979s / AI=1755.43 | 1509.45TF/s / 798.46GB/s / 0.546312s / AI=1890.46 | 1403.20TF/s / 742.25GB/s / 0.587679s / AI=1890.46 | 1449.93TF/s / 766.97GB/s / 0.568739s / AI=1890.46 | 1524.09TF/s / 756.07GB/s / 2.823693s / AI=2015.81 | 1357.90TF/s / 673.63GB/s / 3.169263s / AI=2015.81 | 1381.84TF/s / 685.50GB/s / 3.114371s / AI=2015.81 |
