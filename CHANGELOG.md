# Changelog

本仓是 [Wan-Animate-2](https://github.com/Wan-Video/Wan-Animate-2) 的昇腾适配副本（`BrightenU/Wan-Animate-2-NPU`）。

**每次改代码都要在本文件顶部（`[Unreleased]` 下面）追加一条。** 非琐碎改动再写一篇 `docs/npu/YYYY-MM-DD-短标题.md`。不要改官方 `origin`；推送到 `github`。

格式：日期、为什么、改了什么、怎么验证、数字（延迟 / 误差 / 成片）、还没做。

## [Unreleased]

## 2026-08-29 — 全链路可切换到 4×910B3（64GB）推理

详细说明：[docs/npu/2026-08-29-portable-npu-runtime.md](docs/npu/2026-08-29-portable-npu-runtime.md)

### 为什么

flex 已经换成 split FA。剩下的 CUDA 硬编码（设备、NCCL、autocast、complex RoPE、flash_attn-only）在 910B3 上会直接起不来。需要同一份代码在 H100 和 910B3 之间只改环境变量。

### 改了什么

- `wanxiang/ops/device.py`：`WAN_DEVICE` / 自动探测，`nccl`↔`hccl`，empty_cache / seed / mesh / autocast。
- `wanxiang/models/attention.py`：`flash_attention` 按设备分发（flash-attn / `npu_fusion_attention` / SDPA）。
- `wanxiang/models/wan_animate_2_model.py`：RoPE 改为 float32 `(cos,sin)`；autocast 用 `amp_device_type()`。
- pipeline / eval_i2v / VAE / FSDP mesh / distributed gloo：去掉 `cuda`/`nccl` 写死。
- yaml `sp_size`/`sharding_size`=4，worker cap 到卡数。
- `infer/wan_animate_2_npu_distillation.yaml`、`infer/run_910b3.sh`、`infer/check_npu_port.py`。

### 验证

本机没有 910B3。`infer/check_npu_port.py` 在 CPU/H100 上核对 RoPE 对官方 complex 的误差、设备探测、flash_attention 可跑。910B3 上：`WAN_DEVICE=npu ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 bash infer/run_910b3.sh jobs.json`。

### 未做

- 910B3 实机成片（本机无 NPU）
- 1080P / 3s / ≤40s 延迟（先 640×800 distill 出片）
- 更多 pair 成片率

## 2026-08-29 — 用 split FlashAttention + LSE merge 替换 `flex_attention`

详细说明：[docs/npu/2026-08-29-replace-flex-with-split-fa.md](docs/npu/2026-08-29-replace-flex-with-split-fa.md)

### 为什么

官方 `forward_gen` 自注意力用 CUDA `flex_attention`（inductor）。昇腾没有这条路径。不能整表 SDPA：640×800 会物化 S×S mask 并 OOM。目标是同一套拆分在 H100 上用 FlashAttention，在 910B 上换成 `npu_fusion_attention`。

### 改了什么

- `wanxiang/models/attention.py`：新增 `incontext_attention` / `make_incontext_spec` / `IncontextSpec`；删除 flex/inductor。
- `wanxiang/models/wan_animate_2_model.py`：`forward_gen` 改走 `incontext_attention`。
- `infer/wan_animate_2.yaml`、`infer/wan_animate_2_distillation.yaml`：`sp_size` / `sharding_size` 从 8 改为 2（本机 2×H100）。
- `infer/compare_flex_vs_split.py`：H100 上对官方 flex 做数值 / 延迟对照。
- `infer/compare_a2_batch.py`：成片 smoke 批跑。
- `.gitignore`：忽略 `ckpts` 符号链接和 `infer/outputs_smoke/`。权重不进仓。

### 验证（2×H100）

| 项 | 结果 |
| --- | --- |
| 小/中序列 vs flex | max \|diff\| ~1e-3–6e-3 bf16，cosine > 0.99998 |
| 640×800、H=40、D=128、`log_scale=0` | max \|diff\| **4.88e-4**；flex 42.96 ms/层，split 47.90 ms/层（约 +11%） |
| 同上、`log_scale=-1.3` | split ~49.4 ms/层 |
| 未 compile 的 flex | 27–138GB 易 OOM；split 约 7GB reserved |
| female smoke（distill 10-step，clip_len=49，462×992，16fps） | 成片 `464×928` / 70 帧 / 4.375s；load 112s，generate **109.5s**（原 flex 640×800 generate 108.5s） |

成片路径：`data/compare_out/smoke/female/00000001_3120__female_hello/03_wan_animate_2_npu.mp4`（不在本仓）。

### 未做

- `cuda` / `npu` 设备抽象、NCCL→HCCL、autocast `device_type`
- 910B 上把 flash / CLIP / cross-attn 换成 `npu_fusion_attention`
- RoPE complex64 / float64 → 实数 sin/cos
- 4×910B、`sp_size=4`、1080P / 3s / ≤40s
- 更多 pair 的成片率对照（girl 等）
