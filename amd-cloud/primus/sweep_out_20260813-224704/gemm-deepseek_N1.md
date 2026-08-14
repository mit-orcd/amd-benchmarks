# DeepSeek GEMM Benchmark Report

- Model: Custom
- Date: 2026-08-13 22:54:46
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
| mi355-gpu-33 | 1 | 0 | 848.39TF/s / 1887.14GB/s / 0.022781s / AI=449.56 | 586.47TF/s / 1304.54GB/s / 0.032955s / AI=449.56 | 549.24TF/s / 1221.73GB/s / 0.035189s / AI=449.56 | 921.97TF/s / 2082.08GB/s / 0.074536s / AI=442.81 | 1002.45TF/s / 2263.84GB/s / 0.068551s / AI=442.81 | 1088.46TF/s / 2458.07GB/s / 0.063135s / AI=442.81 | 1480.19TF/s / 903.44GB/s / 0.185705s / AI=1638.40 | 1558.07TF/s / 950.97GB/s / 0.176422s / AI=1638.40 | 1453.95TF/s / 887.42GB/s / 0.189055s / AI=1638.40 | 1563.38TF/s / 890.60GB/s / 0.263734s / AI=1755.43 | 1546.44TF/s / 880.95GB/s / 0.266623s / AI=1755.43 | 1467.06TF/s / 835.73GB/s / 0.281049s / AI=1755.43 | 164.41TF/s / 1003.46GB/s / 0.014695s / AI=163.84 | 195.79TF/s / 1195.02GB/s / 0.012339s / AI=163.84 | 193.25TF/s / 1179.50GB/s / 0.012502s / AI=163.84 | 310.77TF/s / 1795.65GB/s / 0.015548s / AI=173.07 | 257.77TF/s / 1489.38GB/s / 0.018745s / AI=173.07 | 304.51TF/s / 1759.47GB/s / 0.015867s / AI=173.07 | 306.10TF/s / 2540.86GB/s / 0.014031s / AI=120.47 | 264.48TF/s / 2195.39GB/s / 0.016239s / AI=120.47 | 250.34TF/s / 2078.03GB/s / 0.017156s / AI=120.47 | 1501.77TF/s / 855.50GB/s / 0.274555s / AI=1755.43 | 1616.43TF/s / 920.82GB/s / 0.255079s / AI=1755.43 | 1462.55TF/s / 833.16GB/s / 0.281916s / AI=1755.43 | 1546.79TF/s / 818.21GB/s / 0.533126s / AI=1890.46 | 1428.58TF/s / 755.68GB/s / 0.577239s / AI=1890.46 | 1474.80TF/s / 780.13GB/s / 0.559149s / AI=1890.46 | 1563.83TF/s / 775.78GB/s / 2.751927s / AI=2015.81 | 1379.11TF/s / 684.15GB/s / 3.120522s / AI=2015.81 | 1400.97TF/s / 694.99GB/s / 3.071847s / AI=2015.81 |
