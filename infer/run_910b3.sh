#!/usr/bin/env bash
# Launch distill inference on 4× 昇腾 910B3 (64GB).
# Usage:
#   bash infer/run_910b3.sh /path/to/jobs.json
#   WAN_WIDTH=640 WAN_HEIGHT=800 bash infer/run_910b3.sh /path/to/jobs.json
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# CANN env — adjust if the box uses another toolkit path.
for cand in \
  /usr/local/Ascend/ascend-toolkit/set_env.sh \
  /usr/local/Ascend/nnal/atb/set_env.sh \
  "${ASCEND_TOOLKIT_HOME:-}/../set_env.sh"
do
  if [[ -f "$cand" ]]; then
    # shellcheck disable=SC1090
    source "$cand"
    break
  fi
done

export WAN_DEVICE="${WAN_DEVICE:-npu}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-12345}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-3600}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-3600}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# Do not install flash-attn on this box. torch_npu provides npu_fusion_attention.
if [[ -z "${WAN_PYTHON:-}" ]]; then
  if command -v python3 >/dev/null; then
    WAN_PYTHON=python3
  else
    WAN_PYTHON=python
  fi
fi

JOBS="${1:-}"
CONFIG="${2:-$ROOT/infer/wan_animate_2_npu_distillation.yaml}"

if [[ -z "$JOBS" ]]; then
  echo "usage: $0 /path/to/jobs.json [config.yaml]"
  echo "jobs.json is a list of {refer, video, out_dir, width, height, fps, seed, clip_len, step, prompt}"
  exit 2
fi

echo "[910B3] WAN_DEVICE=$WAN_DEVICE devices=$ASCEND_RT_VISIBLE_DEVICES python=$WAN_PYTHON"
echo "[910B3] config=$CONFIG jobs=$JOBS"
"$WAN_PYTHON" "$ROOT/infer/compare_a2_batch.py" --jobs "$JOBS" --config "$CONFIG"
