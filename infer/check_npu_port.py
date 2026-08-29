#!/usr/bin/env python3
"""CPU/H100 checks for the portable NPU port. Does not need torch_npu."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def rope_complex_ref(x, grid_sizes, freqs_c, time_stride=1):
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs_c.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs[0][: f * time_stride : time_stride].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        output.append(torch.cat([x_i, x[i, seq_len:]], dim=0))
    return torch.stack(output).float()


def rope_params_complex(max_seq_len, dim, theta=10000, offset=0):
    freqs = torch.outer(
        torch.arange(max_seq_len) + offset,
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def check_rope():
    from wanxiang.models.wan_animate_2_model import rope_apply, rope_params

    torch.manual_seed(0)
    d = 128
    freqs_r = torch.cat(
        [
            rope_params(64, d - 4 * (d // 6)),
            rope_params(64, 2 * (d // 6)),
            rope_params(64, 2 * (d // 6)),
        ],
        dim=1,
    )
    freqs_c = torch.cat(
        [
            rope_params_complex(64, d - 4 * (d // 6)),
            rope_params_complex(64, 2 * (d // 6)),
            rope_params_complex(64, 2 * (d // 6)),
        ],
        dim=1,
    )
    x = torch.randn(1, 2 * 4 * 6, 4, d)
    grid = torch.tensor([[2, 4, 6]], dtype=torch.long)
    out_r = rope_apply(x, grid, freqs_r)
    out_c = rope_complex_ref(x, grid, freqs_c)
    diff = (out_r - out_c).abs()
    print(f"[rope] max|diff|={diff.max().item():.3e} mean={diff.mean().item():.3e}")
    assert diff.max().item() < 2e-5, diff.max().item()


def check_device():
    from wanxiang.ops.device import amp_device_type, configure_runtime, dist_backend, kind

    configure_runtime()
    print(f"[device] kind={kind()} amp={amp_device_type()} backend={dist_backend()}")
    assert kind() in ("cuda", "npu", "cpu")
    if kind() == "npu":
        assert dist_backend() == "hccl"
        assert amp_device_type() == "npu"
    if kind() == "cuda":
        assert dist_backend() == "nccl"


def check_flash_sdpa():
    from wanxiang.models.attention import flash_attention

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(1, 32, 4, 64, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, 32, 4, 64, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, 32, 4, 64, device=device, dtype=torch.bfloat16)
    out = flash_attention(q, k, v)
    print(f"[flash] device={device} out={tuple(out.shape)} finite={torch.isfinite(out).all().item()}")
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def main():
    check_device()
    check_rope()
    check_flash_sdpa()
    print("OK")


if __name__ == "__main__":
    main()
