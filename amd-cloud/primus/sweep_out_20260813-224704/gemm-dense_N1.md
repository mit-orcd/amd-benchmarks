# Dense GEMM Benchmark Report

- Model: Custom
- Date: 2026-08-13 22:51:58
- Cluster: amd-aig-poolside
- Duration per shape: 5 sec

## Configuration
- mbs: 1
- num_attention_heads: 32
- num_key_value_heads: 32
- head_dim: 128
- hidden_size: 4096
- intermediate_size: 11008
- vocab_size: 32000
- seqlen: 2048
- dtype: bf16

## GEMM Shapes (M, N, K)
- attn_qkv: (2048, 12288, 4096)
- attn_out: (2048, 4096, 4096)
- mlp_up: (2048, 22016, 4096)
- mlp_down: (2048, 4096, 11008)
- vocab: (2048, 32000, 4096)

## Phases
- fwd: forward pass
- wgrad: weight gradient
- dgrad: data gradient

| host | world | rank | attn_out_dgrad | attn_out_fwd | attn_out_wgrad | attn_qkv_dgrad | attn_qkv_fwd | attn_qkv_wgrad | mlp_down_dgrad | mlp_down_fwd | mlp_down_wgrad | mlp_up_dgrad | mlp_up_fwd | mlp_up_wgrad | vocab_dgrad | vocab_fwd | vocab_wgrad |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mi355-gpu-33 | 1 | 0 | 0.056989s / 1205.83TF/s / 1177.57GB/s / AI=1024.00 | 0.052580s / 1306.95TF/s / 1276.32GB/s / AI=1024.00 | 0.052229s / 1315.73TF/s / 1284.89GB/s / AI=1024.00 | 0.180177s / 1144.20TF/s / 931.15GB/s / AI=1228.80 | 0.143188s / 1439.77TF/s / 1171.69GB/s / AI=1228.80 | 0.151264s / 1362.91TF/s / 1109.14GB/s / AI=1228.80 | 0.144707s / 1276.26TF/s / 1050.70GB/s / AI=1214.68 | 0.154469s / 1195.60TF/s / 984.30GB/s / AI=1214.68 | 0.143883s / 1283.57TF/s / 1056.72GB/s / AI=1214.68 | 0.271879s / 1358.57TF/s / 1056.75GB/s / AI=1285.61 | 0.253688s / 1455.99TF/s / 1132.53GB/s / AI=1285.61 | 0.291574s / 1266.81TF/s / 985.38GB/s / AI=1285.61 | 0.383394s / 1400.31TF/s / 1069.38GB/s / AI=1309.46 | 0.364911s / 1471.24TF/s / 1123.54GB/s / AI=1309.46 | 0.398147s / 1348.42TF/s / 1029.75GB/s / AI=1309.46 |
