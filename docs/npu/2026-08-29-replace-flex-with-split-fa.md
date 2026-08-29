# 2026-08-29 用 split FlashAttention + LSE merge 替换 flex_attention

对应提交：`11cc2a3`（本篇是补记，说明为什么这样改、数字从哪来）。

## 背景

官方 Wan-Animate-2 的 `forward_gen` 自注意力是 CUDA `flex_attention`（PyTorch inductor + 自定义 mask / `score_mod`）。CLIP、cross-attn、`forward_ref` 已经走 `flash_attention`。

生产目标是 4×昇腾 910B、CANN / `torch_npu` / HCCL。910B 没有 flex/inductor 这条路。当前验证机是 2×H100：先在 NVIDIA 上把算法和成片率钉住，再换 NPU kernel。

## 为什么不能直接换

`flex_attention` 不是 `flash_attn_varlen_func` 的 drop-in。官方 mask / score 是：

1. 所有有效 driving token 互相可见。
2. query 帧 `t≥1` 还能看到参考帧 `t-1`（`q_frame == ref_frame + 1`）。
3. driving KV 区间 `[hw, 2*hw)` 的 score 加 `log_scale`。distill yaml 是 `-1.3`，base 是 `0.0`。

整表 SDPA + mask 会物化约 S×S，640×800 会 OOM。

## 做法

拆成 2–3 次 FlashAttention，再用 softmax LSE 合并：

- driving 自注意力：`log_scale≈0` 一次；`|log_scale|>0` 时把第二帧 KV 拆出来，LSE 加上 `log_scale` 再 merge。
- 每帧 Q ↔ 对应 reference，再一次 FA。
- CUDA：`flash_attn_func(..., return_attn_probs=True)` → `out [B,S,H,D]`，`lse [B,H,S]`。
- NPU（接口已留，本机未跑）：同样拆分 → `torch_npu.npu_fusion_attention`，用返回的 softmax_max / softmax_sum 还原 LSE。

小序列数值核对可以用 `incontext_attention_dense`（SDPA + 完整 bias），大分辨率不要走这条。

YAML 里 `sp_size` / `sharding_size` 从官方默认 8 改成 2，对齐本机 2×H100。上 4×910B 时改回 4。

## H100 数值 / 延迟（`infer/compare_flex_vs_split.py`）

对照对象：原仓 `/data/zsy/projects/Wan-Animate-2` 的 flex 实现。

| 配置 | 误差 | 延迟 |
| --- | --- | --- |
| 小/中序列 | max \|diff\| ~1e-3–6e-3 bf16，cosine > 0.99998 | — |
| 640×800，clip 打包 `origin_len=37`，hw=2000，q=22016，kv=42112，H=40，D=128，`log_scale=0` | max \|diff\| **4.88e-4** | flex **42.96 ms/层**，split **47.90 ms/层**（约 +11%） |
| 同上，`log_scale=-1.3` | — | split ~**49.4 ms/层** |
| 未 compile 的 flex | — | 峰值显存 27–138GB，易 OOM；split reserved ~7GB |

bf16 下不是 bit-exact，但对生成够用。多一次 FA 换的是可移植性和显存，不是加速。

## Female smoke

- pair：`data/compare_out/smoke/female/00000001_3120__female_hello/pair.json`
- 参考图：`data/refer_images/female/00000001_3120.png`
- 驱动：`data/driven_videos/female/female_hello.mp4`（无音轨，ffmpeg 抽 wav 失败可忽略）
- 配置：`wan_animate_2_distillation.yaml`，`clip_len=49`，`step=10`，`guide=1.0`，`seed=42`，16fps，输入 462×992（pipeline `resize_by_area` + 16 对齐）
- 产出：`464×928`，70 帧，4.375s @16fps
- 耗时：load **112s**，generate **109.5s**（原 CUDA flex、640×800 generate 是 108.5s）
- 成片：`data/compare_out/smoke/female/00000001_3120__female_hello/03_wan_animate_2_npu.mp4` 以及 `npu_split_fa/03_wan_animate_2.mp4`

`compare_a2_batch.py` 若 `out_dir/03_wan_animate_2.mp4` 已存在且 >1000 字节会跳过，重跑要换 `out_dir`。

## 复跑 smoke

```bash
export PYTHONPATH=/data/zsy/projects/Wan-Animate-2-NPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1
/data/anaconda3/envs/wan_animate_2/bin/python \
  /data/zsy/projects/Wan-Animate-2-NPU/infer/compare_a2_batch.py \
  --jobs /data/zsy/projects/data/compare_out/smoke/female/00000001_3120__female_hello/_a2_npu_job.json \
  --config ./wan_animate_2_distillation.yaml
```

conda：`/data/anaconda3/envs/wan_animate_2`（torch 2.7.1+cu126，flash_attn 2.8.3）。`ckpts` 是指向 `../Wan-Animate-2/ckpts` 的符号链接，不要复制、不要提交。

## 后续（不要在本条里混做）

1. 设备抽象：`cuda`/`npu`，NCCL→HCCL，autocast `device_type`。
2. 910B：flash / CLIP / cross-attn → `npu_fusion_attention`（不要用 LLM infer 算子）。
3. RoPE：complex64 / float64 → 实数 sin/cos。
4. 4×910B、`sp_size=4`、1080P / 3s / ≤40s。
5. 更多 smoke（girl 等）对成片率。
