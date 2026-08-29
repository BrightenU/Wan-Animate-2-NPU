"""cuda / npu device switch for Wan-Animate-2-NPU.

Detection order:
  1. WAN_DEVICE=cuda|npu|cpu
  2. torch.npu.is_available() after importing torch_npu
  3. torch.cuda.is_available()
  4. cpu (import / unit tests only)

On 910B3: source CANN, set WAN_DEVICE=npu (or rely on auto-detect),
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3. Do not install flash-attn on NPU.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

_KIND = None
_NPU_IMPORTED = False


def _try_import_npu():
    global _NPU_IMPORTED
    if _NPU_IMPORTED:
        return True
    try:
        import torch_npu  # noqa: F401
        _NPU_IMPORTED = True
        return True
    except Exception:
        return False


def _npu_is_available():
    if not _try_import_npu():
        return False
    npu = getattr(torch, "npu", None)
    return npu is not None and npu.is_available()


def detect_kind():
    forced = os.getenv("WAN_DEVICE", "").strip().lower()
    if forced in ("cuda", "npu", "cpu"):
        if forced == "npu" and not _npu_is_available():
            raise RuntimeError(
                "WAN_DEVICE=npu but torch_npu is not available. "
                "Install the CANN + torch_npu wheel matching this 910B3."
            )
        return forced
    if _npu_is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def kind():
    global _KIND
    if _KIND is None:
        _KIND = detect_kind()
    return _KIND


def reset_kind():
    """Tests only."""
    global _KIND
    _KIND = None


def is_npu():
    return kind() == "npu"


def is_cuda():
    return kind() == "cuda"


def amp_device_type():
    k = kind()
    return "cpu" if k == "cpu" else k


def dist_backend():
    k = kind()
    if k == "npu":
        return "hccl"
    if k == "cuda":
        return "nccl"
    return "gloo"


def device_string(local_rank=0):
    k = kind()
    if k == "cpu":
        return "cpu"
    return f"{k}:{int(local_rank)}"


def as_device(local_rank=0):
    return torch.device(device_string(local_rank))


def _mod():
    k = kind()
    if k == "npu":
        _try_import_npu()
        return torch.npu
    if k == "cuda":
        return torch.cuda
    return None


def is_available():
    m = _mod()
    return True if m is None else bool(m.is_available())


def device_count():
    m = _mod()
    if m is None:
        return 1
    try:
        return int(m.device_count())
    except Exception:
        return 1


def set_device(local_rank):
    m = _mod()
    if m is not None:
        m.set_device(int(local_rank))


def current_device():
    m = _mod()
    if m is None:
        return torch.device("cpu")
    return torch.device(f"{kind()}:{m.current_device()}")


def empty_cache():
    m = _mod()
    if m is not None:
        m.empty_cache()


def synchronize():
    m = _mod()
    if m is not None:
        m.synchronize()


def memory_allocated():
    m = _mod()
    return float(m.memory_allocated()) if m is not None else 0.0


def memory_reserved():
    m = _mod()
    return float(m.memory_reserved()) if m is not None else 0.0


def manual_seed_all(seed):
    torch.manual_seed(int(seed))
    m = _mod()
    if m is not None and hasattr(m, "manual_seed_all"):
        m.manual_seed_all(int(seed))


def autocast(dtype=torch.bfloat16, enabled=True):
    return torch.autocast(device_type=amp_device_type(), dtype=dtype, enabled=enabled)


def barrier(device_id=None, **kwargs):
    if not (dist.is_available() and dist.is_initialized()):
        return
    backend = dist.get_backend()
    if device_id is not None and backend == "nccl":
        dist.barrier(device_ids=[int(device_id)], **kwargs)
        return
    dist.barrier(**kwargs)


def configure_runtime():
    """Allocator + NPU plugin. Safe to call more than once."""
    k = kind()
    if k == "cuda":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    elif k == "npu":
        _try_import_npu()
        os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "3600")
        os.environ.setdefault("HCCL_EXEC_TIMEOUT", "3600")
    return k
