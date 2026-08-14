# Dense GEMM Benchmark Report

- Model: Custom
- Date: 2026-08-13 23:02:03
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
| mi355-gpu-33 | 3 | 0 | 0.056931s / 1207.07TF/s / 1178.78GB/s / AI=1024.00 | 0.052837s / 1300.60TF/s / 1270.12GB/s / AI=1024.00 | 0.052461s / 1309.92TF/s / 1279.22GB/s / AI=1024.00 | 0.180285s / 1143.51TF/s / 930.59GB/s / AI=1228.80 | 0.144039s / 1431.27TF/s / 1164.77GB/s / AI=1228.80 | 0.151625s / 1359.66TF/s / 1106.49GB/s / AI=1228.80 | 0.142782s / 1293.46TF/s / 1064.86GB/s / AI=1214.68 | 0.157138s / 1175.29TF/s / 967.58GB/s / AI=1214.68 | 0.145368s / 1270.45TF/s / 1045.92GB/s / AI=1214.68 | 0.274986s / 1343.22TF/s / 1044.82GB/s / AI=1285.61 | 0.254186s / 1453.14TF/s / 1130.31GB/s / AI=1285.61 | 0.288900s / 1278.53TF/s / 994.50GB/s / AI=1285.61 | 0.383241s / 1400.87TF/s / 1069.81GB/s / AI=1309.46 | 0.365543s / 1468.69TF/s / 1121.60GB/s / AI=1309.46 | 0.399178s / 1344.94TF/s / 1027.09GB/s / AI=1309.46 |
| mi355-gpu-33 | 3 | 1 | 0.057981s / 1185.20TF/s / 1157.42GB/s / AI=1024.00 | 0.053320s / 1288.80TF/s / 1258.60GB/s / AI=1024.00 | 0.052998s / 1296.63TF/s / 1266.24GB/s / AI=1024.00 | 0.171031s / 1205.39TF/s / 980.95GB/s / AI=1228.80 | 0.145473s / 1417.16TF/s / 1153.29GB/s / AI=1228.80 | 0.153978s / 1338.88TF/s / 1089.59GB/s / AI=1228.80 | 0.144458s / 1278.46TF/s / 1052.51GB/s / AI=1214.68 | 0.160771s / 1148.74TF/s / 945.72GB/s / AI=1214.68 | 0.149032s / 1239.22TF/s / 1020.20GB/s / AI=1214.68 | 0.279240s / 1322.76TF/s / 1028.90GB/s / AI=1285.61 | 0.258650s / 1428.06TF/s / 1110.80GB/s / AI=1285.61 | 0.287717s / 1283.78TF/s / 998.58GB/s / AI=1285.61 | 0.391210s / 1372.33TF/s / 1048.01GB/s / AI=1309.46 | 0.370139s / 1450.46TF/s / 1107.67GB/s / AI=1309.46 | 0.405691s / 1323.35TF/s / 1010.60GB/s / AI=1309.46 |
| mi355-gpu-33 | 3 | 2 | 0.057097s / 1203.56TF/s / 1175.35GB/s / AI=1024.00 | 0.052293s / 1314.13TF/s / 1283.33GB/s / AI=1024.00 | 0.052303s / 1313.88TF/s / 1283.09GB/s / AI=1024.00 | 0.183658s / 1122.51TF/s / 913.50GB/s / AI=1228.80 | 0.143931s / 1432.34TF/s / 1165.64GB/s / AI=1228.80 | 0.152872s / 1348.57TF/s / 1097.47GB/s / AI=1228.80 | 0.146159s / 1263.58TF/s / 1040.26GB/s / AI=1214.68 | 0.159053s / 1161.14TF/s / 955.93GB/s / AI=1214.68 | 0.150643s / 1225.97TF/s / 1009.30GB/s / AI=1214.68 | 0.276277s / 1336.94TF/s / 1039.93GB/s / AI=1285.61 | 0.265347s / 1392.02TF/s / 1082.77GB/s / AI=1285.61 | 0.285210s / 1295.07TF/s / 1007.36GB/s / AI=1285.61 | 0.386687s / 1388.39TF/s / 1060.27GB/s / AI=1309.46 | 0.366765s / 1463.80TF/s / 1117.86GB/s / AI=1309.46 | 0.411177s / 1305.69TF/s / 997.12GB/s / AI=1309.46 |
