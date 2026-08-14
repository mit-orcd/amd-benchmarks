# Dense GEMM Benchmark Report

- Model: Custom
- Date: 2026-08-13 22:56:57
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
| mi355-gpu-33 | 2 | 0 | 0.056752s / 1210.87TF/s / 1182.49GB/s / AI=1024.00 | 0.052523s / 1308.36TF/s / 1277.69GB/s / AI=1024.00 | 0.052160s / 1317.48TF/s / 1286.60GB/s / AI=1024.00 | 0.173490s / 1188.30TF/s / 967.04GB/s / AI=1228.80 | 0.143602s / 1435.62TF/s / 1168.31GB/s / AI=1228.80 | 0.151550s / 1360.33TF/s / 1107.04GB/s / AI=1228.80 | 0.142044s / 1300.18TF/s / 1070.39GB/s / AI=1214.68 | 0.161156s / 1145.99TF/s / 943.46GB/s / AI=1214.68 | 0.144188s / 1280.85TF/s / 1054.48GB/s / AI=1214.68 | 0.275812s / 1339.20TF/s / 1041.69GB/s / AI=1285.61 | 0.253727s / 1455.77TF/s / 1132.36GB/s / AI=1285.61 | 0.286805s / 1287.87TF/s / 1001.76GB/s / AI=1285.61 | 0.382782s / 1402.55TF/s / 1071.09GB/s / AI=1309.46 | 0.364857s / 1471.46TF/s / 1123.71GB/s / AI=1309.46 | 0.398903s / 1345.87TF/s / 1027.80GB/s / AI=1309.46 |
| mi355-gpu-33 | 2 | 1 | 0.057645s / 1192.12TF/s / 1164.18GB/s / AI=1024.00 | 0.053510s / 1284.23TF/s / 1254.14GB/s / AI=1024.00 | 0.052930s / 1298.32TF/s / 1267.89GB/s / AI=1024.00 | 0.170524s / 1208.97TF/s / 983.87GB/s / AI=1228.80 | 0.145388s / 1417.99TF/s / 1153.96GB/s / AI=1228.80 | 0.153755s / 1340.82TF/s / 1091.16GB/s / AI=1228.80 | 0.144241s / 1280.38TF/s / 1054.09GB/s / AI=1214.68 | 0.158978s / 1161.69TF/s / 956.38GB/s / AI=1214.68 | 0.147745s / 1250.01TF/s / 1029.09GB/s / AI=1214.68 | 0.276953s / 1333.68TF/s / 1037.40GB/s / AI=1285.61 | 0.258445s / 1429.19TF/s / 1111.69GB/s / AI=1285.61 | 0.286805s / 1287.87TF/s / 1001.76GB/s / AI=1285.61 | 0.395435s / 1357.67TF/s / 1036.81GB/s / AI=1309.46 | 0.369937s / 1451.25TF/s / 1108.28GB/s / AI=1309.46 | 0.405256s / 1324.77TF/s / 1011.69GB/s / AI=1309.46 |
