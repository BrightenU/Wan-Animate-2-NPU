#!/usr/bin/env python3
"""H100 A/B: original flex_attention vs split-FA + LSE merge."""
import importlib.util
import sys
import time
from pathlib import Path

import torch
from torch.nn.attention.flex_attention import create_block_mask

ROOT = Path(__file__).resolve().parents[1]
ORIG = Path("/data/zsy/projects/Wan-Animate-2")
sys.path.insert(0, str(ROOT))

from wanxiang.models.attention import (  # noqa: E402
    build_incontext_attn_bias,
    incontext_attention,
    incontext_attention_dense,
    make_incontext_spec,
)


def load_orig_flex():
    spec = importlib.util.spec_from_file_location(
        "orig_a2_attention", ORIG / "wanxiang/models/attention.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.flex_attention


def orig_block_mask(spec, device):
    hw, q_limit, k_limit, q_total = spec.hw, spec.q_limit, spec.k_limit, spec.q_total

    def attention_mask_logic(b, h, q_idx, kv_idx):
        q_valid = q_idx < q_limit
        is_base_attention = kv_idx < q_limit
        q_frame = q_idx // hw
        is_first_part = kv_idx < q_total
        kv_frame_1 = kv_idx // hw
        kv_is_valid_1 = kv_idx < q_limit
        rel_kv_idx = kv_idx - q_total
        kv_frame_2 = (rel_kv_idx // hw) + 1
        kv_is_valid_2 = rel_kv_idx < k_limit
        kv_frame = torch.where(is_first_part, kv_frame_1, kv_frame_2)
        kv_is_valid = torch.where(is_first_part, kv_is_valid_1, kv_is_valid_2)
        is_cond_attention = (q_frame == kv_frame) & kv_is_valid
        return q_valid & (is_base_attention | is_cond_attention)

    return create_block_mask(
        attention_mask_logic,
        B=None,
        H=None,
        Q_LEN=spec.q_len_total,
        KV_LEN=spec.k_len_total,
        device=device,
        _compile=True,
    )


def orig_score_mod(hw, log_scale):
    # Bake as defaults so inductor does not capture a 0-d tensor (.item() breaks compile).
    def _mod(score, b_idx, h_idx, q_idx, kv_idx, _hw=int(hw), _ls=float(log_scale)):
        condition = (kv_idx >= _hw) & (kv_idx < (2 * _hw))
        return torch.where(condition, score + _ls, score)

    return _mod


def stats(a, b, name):
    d = (a.float() - b.float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0
    )
    print(
        f"  {name}: max={d.max().item():.5e} mean={d.mean().item():.5e} "
        f"cos={cos.item():.8f}"
    )
    return d.max().item()


def bench(fn, n_warm=5, n_run=20):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_run):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_run * 1000.0


def run_case(origin_len, origin_area, log_scale, heads, dim, orig_flex, tag):
    device = "cuda"
    dtype = torch.bfloat16
    spec = make_incontext_spec(origin_len, origin_area, log_scale)
    B = 1
    print(
        f"\n=== {tag} origin_len={origin_len} area={origin_area} log_scale={log_scale} "
        f"hw={spec.hw} q={spec.q_len_total} kv={spec.k_len_total} H={heads} D={dim} ==="
    )
    torch.manual_seed(0)
    q = torch.randn(B, spec.q_len_total, heads, dim, device=device, dtype=dtype)
    k = torch.randn(B, spec.k_len_total, heads, dim, device=device, dtype=dtype)
    v = torch.randn(B, spec.k_len_total, heads, dim, device=device, dtype=dtype)

    out_split = incontext_attention(q, k, v, spec=spec)

    use_dense = spec.q_len_total * spec.k_len_total * heads < 8_000_000
    out_dense = None
    if use_dense:
        out_dense = incontext_attention_dense(q, k, v, spec)
        stats(out_split, out_dense, "split vs dense-sdpa")

    max_diff = None
    ms_flex = None
    try:
        mask = orig_block_mask(spec, device)
        score_mod = orig_score_mod(spec.hw, spec.log_scale)
        flex_impl = orig_flex
        try:
            out_flex = flex_impl(
                q, k, v, block_mask=mask, kernel_options=None, score_mod=score_mod
            )
        except Exception:
            from torch.nn.attention.flex_attention import flex_attention as flex_raw

            def flex_impl(q, k, v, block_mask=None, kernel_options=None, score_mod=None, dtype=torch.bfloat16):
                x = flex_raw(
                    query=q.transpose(2, 1),
                    key=k.transpose(2, 1),
                    value=v.transpose(2, 1),
                    block_mask=block_mask,
                    kernel_options=kernel_options,
                    score_mod=score_mod,
                ).transpose(2, 1)
                return x.type(q.dtype)

            out_flex = flex_impl(
                q, k, v, block_mask=mask, kernel_options=None, score_mod=score_mod
            )
            print("  (using uncompiled flex_attention)")
        max_diff = stats(out_split, out_flex, "split vs flex")
        ms_flex = bench(
            lambda: flex_impl(q, k, v, block_mask=mask, kernel_options=None, score_mod=score_mod)
        )
    except Exception as e:
        print(f"  flex skipped: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
        if out_dense is not None:
            max_diff = (out_split.float() - out_dense.float()).abs().max().item()

    ms_split = bench(lambda: incontext_attention(q, k, v, spec=spec))
    if ms_flex:
        print(
            f"  latency  flex={ms_flex:.2f} ms  split-fa={ms_split:.2f} ms  "
            f"split/flex={ms_split / ms_flex:.2f}x"
        )
    else:
        print(f"  latency  split-fa={ms_split:.2f} ms  (flex unavailable)")
    return max_diff, ms_flex, ms_split


def main():
    orig_flex = load_orig_flex()
    print("loaded original flex_attention (inductor compile on first use)")

    cases = [
        # tiny: dense SDPA also fits; flex block size 128
        dict(origin_len=4, origin_area=(128, 256), log_scale=0.0, heads=4, dim=64, tag="tiny log0"),
        dict(origin_len=4, origin_area=(128, 256), log_scale=-1.3, heads=4, dim=64, tag="tiny log-1.3"),
        # closer to 480P packing (hw=2000 at 640x800 is heavy; use smaller spatial)
        dict(origin_len=17, origin_area=(256, 256), log_scale=-1.3, heads=8, dim=128, tag="mid 256^2"),
        # yaml 640x800 clip_len=37, but fewer heads to keep flex compile reasonable
        dict(origin_len=37, origin_area=(640, 800), log_scale=0.0, heads=8, dim=128, tag="640x800 H8 log0"),
        dict(origin_len=37, origin_area=(640, 800), log_scale=-1.3, heads=8, dim=128, tag="640x800 H8"),
        dict(origin_len=37, origin_area=(640, 800), log_scale=-1.3, heads=40, dim=128, tag="640x800 Wan H40"),
    ]

    rows = []
    for c in cases:
        torch.cuda.empty_cache()
        try:
            rows.append((c["tag"], *run_case(orig_flex=orig_flex, **{k: v for k, v in c.items()})))
        except Exception as e:
            print(f"\n=== {c['tag']} FAILED: {type(e).__name__}: {e} ===")
            rows.append((c["tag"], None, None, None))

    print("\n==== summary ====")
    for tag, diff, flex_ms, split_ms in rows:
        diff_s = "n/a" if diff is None else f"{diff:.4e}"
        flex_s = "n/a" if flex_ms is None else f"{flex_ms:.2f}ms"
        split_s = "n/a" if split_ms is None else f"{split_ms:.2f}ms"
        extra = ""
        if flex_ms and split_ms:
            extra = f"  ({split_ms / flex_ms:.2f}x)"
        print(f"{tag}: max|diff|={diff_s}  flex={flex_s}  split={split_s}{extra}")


if __name__ == "__main__":
    main()
