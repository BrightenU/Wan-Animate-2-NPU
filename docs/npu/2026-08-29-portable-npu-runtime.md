# 2026-08-29 全链路可切换到 910B3 的运行时

对应产品目标：4×昇腾 910B3（64GB）上直接推理。本机仍是 2×H100，用来验证算法；910B3 上设 `WAN_DEVICE=npu` 即可。

## 通篇检查清单（已改）

| # | 阻塞点 | 位置 | 做法 |
| --- | --- | --- | --- |
| 1 | `flex_attention` / inductor | `forward_gen` | 上一提交：split FA + LSE merge |
| 2 | `flash_attn` 仅 CUDA | CLIP / cross-attn / `forward_ref` | CUDA 走 flash-attn；NPU 走 `npu_fusion_attention`；否则 SDPA |
| 3 | `device="cuda"` / `torch.cuda.*` | pipeline、VAE、eval、base | `wanxiang/ops/device.py`，由 `WAN_DEVICE` 或自动探测切换 |
| 4 | `dist.init_process_group(backend="nccl")` | pipeline worker | `nccl` / `hccl` / `gloo` |
| 5 | `init_device_mesh(device_type='cuda')` | FSDP | `device_kind()` |
| 6 | `autocast(device_type='cuda')` | model / pipeline / CLIP | `amp_device_type()` → `cuda` 或 `npu` |
| 7 | RoPE `complex64` + `float64` | `rope_params` / `rope_apply` | 实数 `(cos, sin)`，float32 |
| 8 | `distributed.py` 只认 nccl | gloo 组 / pickle gather | 允许 `hccl`，device=`npu` |
| 9 | 卡数写死 2 或 8 | yaml | `sp_size=4` / `sharding_size=4`，worker 再 cap 到 `world_size` |
| 10 | 启动脚本 | — | `infer/run_910b3.sh` + `infer/wan_animate_2_npu_distillation.yaml` |

未改（本机 H100 对照脚本，不进 910B 推理路径）：`infer/compare_flex_vs_split.py`。

## 910B3 64GB ×4 怎么跑

```bash
# 1. CANN + torch_npu（与 CANN 版本匹配的 wheel）。不要装 flash-attn。
# 2. ckpts 仍用符号链接：ln -s /path/to/Wan-Animate-2/ckpts ckpts
# 3. 四卡可见
export WAN_DEVICE=npu
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
bash infer/run_910b3.sh /path/to/jobs.json
```

`jobs.json` 与 H100 批跑相同。worker 用 `mp.spawn` 起 4 个进程，HCCL 组 world_size=4，FSDP mesh `(1, 4)`，序列并行 4。

内存：DiT ~28GB bf16 权重，4 卡 FULL/HYBRID shard 后约 7GB/卡，64GB 留给激活。先跑 640×800 或 female 462×992、distill 10-step。1080P / 3s 是产品目标，第一刀先确认能出片再抬分辨率。

## 环境变量

| 变量 | 含义 |
| --- | --- |
| `WAN_DEVICE` | `npu` / `cuda` / `cpu`。不设则自动探测 |
| `ASCEND_RT_VISIBLE_DEVICES` | NPU 卡号 |
| `RANK` / `WORLD_SIZE` | 机器级 PMI，单机保持 0 / 1 |
| `MASTER_ADDR` / `MASTER_PORT` | 进程组 |
| `HCCL_CONNECT_TIMEOUT` / `HCCL_EXEC_TIMEOUT` | 默认 3600 |
| `PYTORCH_NPU_ALLOC_CONF` | 默认 `expandable_segments:True` |

## 本机验证

`infer/check_npu_port.py`：设备探测、RoPE 对官方 complex 的误差、flash_attention 可跑。
