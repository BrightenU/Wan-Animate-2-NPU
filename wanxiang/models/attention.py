import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    from flash_attn_interface import flash_attn_varlen_func

    FLASH_VER = 3
except ModuleNotFoundError:
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func

        FLASH_VER = 2
    except ModuleNotFoundError:
        flash_attn_func = None
        flash_attn_varlen_func = None
        FLASH_VER = None

print(f"[PreInfo] Use flash attention={FLASH_VER}")

__all__ = [
    "flash_attention",
    "incontext_attention",
    "make_incontext_spec",
]


@dataclass(frozen=True)
class IncontextSpec:
    hw: int
    q_limit: int
    k_limit: int
    q_total: int
    q_len_total: int
    k_len_total: int
    log_scale: float


def make_incontext_spec(origin_len, origin_area, log_scale=0.0):
    """Same packing as the original flex create_mask."""
    origin_latent_f = int(origin_len) // 4 + 1
    if torch.is_tensor(origin_area):
        hw = int(origin_area.prod().item() // 256)
    else:
        hw = int(origin_area[0] * origin_area[1] // 256)
    q_len = (origin_latent_f + 1) * hw
    k_len = origin_latent_f * hw
    q_len_total = math.ceil(q_len / 128) * 128
    k_extra_len_total = math.ceil(k_len / 128) * 128
    return IncontextSpec(
        hw=hw,
        q_limit=q_len,
        k_limit=k_len,
        q_total=q_len_total,
        q_len_total=q_len_total,
        k_len_total=q_len_total + k_extra_len_total,
        log_scale=float(log_scale),
    )


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(
            device=q.device, non_blocking=True
        )
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(
            device=k.device, non_blocking=True
        )
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if FLASH_VER == 3:
        x = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )[0].unflatten(0, (b, lq))
    else:
        assert FLASH_VER == 2
        x = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))

    return x.type(out_dtype)


def _math_out_lse(q, k, v, softmax_scale=None):
    scale = (q.shape[-1] ** -0.5) if softmax_scale is None else softmax_scale
    qh = q.float().transpose(1, 2)
    kh = k.float().transpose(1, 2)
    vh = v.float().transpose(1, 2)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
    lse = torch.logsumexp(scores, dim=-1)
    out = torch.matmul(torch.softmax(scores, dim=-1), vh)
    return out.transpose(1, 2).type_as(q), lse


def _npu_out_lse(q, k, v, softmax_scale=None):
    import torch_npu

    scale = (q.shape[-1] ** -0.5) if softmax_scale is None else softmax_scale
    qn = q.transpose(1, 2).contiguous()
    kn = k.transpose(1, 2).contiguous()
    vn = v.transpose(1, 2).contiguous()
    ret = torch_npu.npu_fusion_attention(
        qn, kn, vn, qn.shape[1], "BNSD", scale=scale, keep_prob=1.0
    )
    out = ret[0].transpose(1, 2)
    softmax_max, softmax_sum = ret[1], ret[2]
    lse = softmax_max.float() + torch.log(softmax_sum.float().clamp_min(1e-12))
    return out.type_as(q), lse


def _attn_out_lse(q, k, v, softmax_scale=None):
    """
    q/k/v: [B, S, H, D]
    returns out [B, S_q, H, D], lse [B, H, S_q] float32
    """
    if q.numel() == 0 or k.numel() == 0:
        B, Lq, H, D = q.shape
        return q.new_zeros(B, Lq, H, D), q.new_full((B, H, Lq), float("-inf"), dtype=torch.float32)

    if q.device.type == "cuda" and FLASH_VER == 2 and flash_attn_func is not None:
        out, lse, _ = flash_attn_func(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            softmax_scale=softmax_scale,
            return_attn_probs=True,
        )
        return out, lse.float()

    if q.device.type == "npu":
        return _npu_out_lse(q, k, v, softmax_scale=softmax_scale)

    return _math_out_lse(q, k, v, softmax_scale=softmax_scale)


def _pad_out_lse(out, lse, seq_len):
    B, L, H, D = out.shape
    if L == seq_len:
        return out, lse
    out_p = out.new_zeros(B, seq_len, H, D)
    lse_p = lse.new_full((B, H, seq_len), float("-inf"))
    out_p[:, :L] = out
    lse_p[:, :, :L] = lse
    return out_p, lse_p


def _place_out_lse(out_src, lse_src, seq_len, start):
    B, L, H, D = out_src.shape
    out_p = out_src.new_zeros(B, seq_len, H, D)
    lse_p = lse_src.new_full((B, H, seq_len), float("-inf"))
    end = start + L
    out_p[:, start:end] = out_src
    lse_p[:, :, start:end] = lse_src
    return out_p, lse_p


def _merge_lse_outs(parts):
    lse_stack = torch.stack([p[1] for p in parts], dim=0)
    finite = torch.isfinite(lse_stack)
    fill = lse_stack.new_tensor(-1e30)
    safe = torch.where(finite, lse_stack, fill)
    m = safe.amax(dim=0)
    w = torch.where(finite, torch.exp(lse_stack - m.unsqueeze(0)), torch.zeros_like(lse_stack))
    w_sum = w.sum(dim=0).clamp_min(1e-12)
    out_stack = torch.stack([p[0].float() for p in parts], dim=0)
    ww = w.permute(0, 1, 3, 2).unsqueeze(-1)
    merged = (out_stack * ww).sum(dim=0) / w_sum.permute(0, 2, 1).unsqueeze(-1)
    none = ~torch.isfinite(m)
    if none.any():
        merged = merged.masked_fill(none.permute(0, 2, 1).unsqueeze(-1), 0)
    return merged.type_as(parts[0][0])


def build_incontext_attn_bias(spec: IncontextSpec, device, dtype):
    """Additive SDPA mask: 0 attend, -inf blocked, plus log_scale on driving frame 1."""
    q_idx = torch.arange(spec.q_len_total, device=device)
    kv_idx = torch.arange(spec.k_len_total, device=device)
    q = q_idx[:, None]
    kv = kv_idx[None, :]
    hw = spec.hw
    q_valid = q < spec.q_limit
    is_base = kv < spec.q_limit
    q_frame = torch.div(q, hw, rounding_mode="floor")
    is_first_part = kv < spec.q_total
    kv_frame_1 = torch.div(kv, hw, rounding_mode="floor")
    rel_kv = kv - spec.q_total
    kv_frame_2 = torch.div(rel_kv, hw, rounding_mode="floor") + 1
    kv_is_valid_1 = kv < spec.q_limit
    kv_is_valid_2 = rel_kv < spec.k_limit
    kv_frame = torch.where(is_first_part, kv_frame_1, kv_frame_2)
    kv_is_valid = torch.where(is_first_part, kv_is_valid_1, kv_is_valid_2)
    is_cond = (q_frame == kv_frame) & kv_is_valid
    allow = q_valid & (is_base | is_cond)
    bias = torch.zeros(spec.q_len_total, spec.k_len_total, device=device, dtype=dtype)
    bias = bias.masked_fill(~allow, float("-inf"))
    if spec.log_scale != 0.0:
        second = (kv_idx >= hw) & (kv_idx < 2 * hw)
        bias = bias + spec.log_scale * second.to(dtype)
    return bias


def incontext_attention_dense(q, k, v, spec: IncontextSpec, dtype=torch.bfloat16):
    """Dense SDPA with the exact flex mask. Only for small sequences / numerical checks."""
    half_dtypes = (torch.float16, torch.bfloat16)
    out_dtype = q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    q, k, v = half(q), half(k), half(v)
    q = q.to(v.dtype)
    k = k.to(v.dtype)
    bias = build_incontext_attn_bias(spec, q.device, q.dtype)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=bias,
        dropout_p=0.0,
    )
    return out.transpose(1, 2).contiguous().type(out_dtype)


def incontext_attention(
    q,
    k,
    v,
    origin_len=None,
    origin_area=None,
    log_scale=0.0,
    spec: IncontextSpec = None,
    dtype=torch.bfloat16,
):
    """
    Replace flex_attention in forward_gen.

    Same mask:
      - every valid query attends to all valid driving tokens
      - query frame t>=1 also attends to reference frame t-1
    Same score_mod: driving tokens in [hw, 2*hw) get `log_scale` added to the score.

    Implemented as 2–3 FlashAttention (or NPU fusion) calls + log-sum-exp merge,
    so it does not materialize the SxS map. Maps to npu_fusion_attention on 910B.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    out_dtype = q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    q, k, v = half(q), half(k), half(v)
    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if spec is None:
        spec = make_incontext_spec(origin_len, origin_area, log_scale)

    hw = spec.hw
    q_limit = spec.q_limit
    k_limit = spec.k_limit
    q_total = spec.q_total
    seq_len = spec.q_len_total
    log_scale = spec.log_scale

    q_drv = q[:, :q_limit]
    k_drv = k[:, :q_limit]
    v_drv = v[:, :q_limit]

    parts = []
    split_scale = abs(log_scale) > 1e-8 and (2 * hw) <= q_limit and hw > 0

    if split_scale:
        k_rest = torch.cat([k_drv[:, :hw], k_drv[:, 2 * hw :]], dim=1)
        v_rest = torch.cat([v_drv[:, :hw], v_drv[:, 2 * hw :]], dim=1)
        out_a, lse_a = _attn_out_lse(q_drv, k_rest, v_rest)
        parts.append(_pad_out_lse(out_a, lse_a, seq_len))

        out_b, lse_b = _attn_out_lse(q_drv, k_drv[:, hw:2 * hw], v_drv[:, hw:2 * hw])
        lse_b = lse_b + log_scale
        parts.append(_pad_out_lse(out_b, lse_b, seq_len))
    else:
        out_a, lse_a = _attn_out_lse(q_drv, k_drv, v_drv)
        parts.append(_pad_out_lse(out_a, lse_a, seq_len))

    if k_limit > 0 and q_limit > hw:
        n_ref_frames = k_limit // hw
        q_cond = q[:, hw : hw + n_ref_frames * hw]
        k_cond = k[:, q_total : q_total + n_ref_frames * hw]
        v_cond = v[:, q_total : q_total + n_ref_frames * hw]
        B, _, H, D = q_cond.shape
        q_cond = q_cond.reshape(B * n_ref_frames, hw, H, D)
        k_cond = k_cond.reshape(B * n_ref_frames, hw, H, D)
        v_cond = v_cond.reshape(B * n_ref_frames, hw, H, D)
        out_c, lse_c = _attn_out_lse(q_cond, k_cond, v_cond)
        out_c = out_c.reshape(B, n_ref_frames * hw, H, D)
        lse_c = lse_c.reshape(B, n_ref_frames, H, hw).permute(0, 2, 1, 3).reshape(B, H, n_ref_frames * hw)
        parts.append(_place_out_lse(out_c, lse_c, seq_len, start=hw))

    x = _merge_lse_outs(parts)
    return x.type(out_dtype)
