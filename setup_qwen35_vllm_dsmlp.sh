#!/usr/bin/env bash
set -euo pipefail

# Run this INSIDE your GPU pod/container, not on dsmlp-login.
#
# Typical flow from dsmlp-login:
#   IDENTITY_PROXY_PORTS=1 launch.sh -g 1 -b
#   kubectl get pod
#   kubesh <your-pod-name>
# Then inside the pod:
#   bash setup_qwen35_vllm_dsmlp_fixed.sh
#
# Optional overrides:
#   ENV_NAME=vllm-qwen35 PYTHON_VERSION=3.10 MODEL_ID=Qwen/Qwen3.5-4B \
#   MODEL_DIR=/private/home/$USER/private/models/Qwen3.5-4B \
#   HF_HOME=/private/home/$USER/private/hf_cache \
#   PORT=8000 MAX_MODEL_LEN=512 GPU_MEMORY_UTILIZATION=0.80 \
#   bash setup_qwen35_vllm_dsmlp_fixed.sh

ENV_NAME="${ENV_NAME:-vllm-qwen35}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-4B}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
LOG_DIR="${LOG_DIR:-$HOME/vllm-logs}"

# Prefer /private if available, to avoid filling ~ quota.
PRIVATE_ROOT="/private/home/$USER/private"
if [[ -d "$PRIVATE_ROOT" ]]; then
  DEFAULT_STORAGE_ROOT="$PRIVATE_ROOT"
else
  DEFAULT_STORAGE_ROOT="$HOME"
fi

HF_HOME="${HF_HOME:-$DEFAULT_STORAGE_ROOT/hf_cache}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
MODEL_DIR="${MODEL_DIR:-$DEFAULT_STORAGE_ROOT/models/$(basename "$MODEL_ID")}"

LOG_FILE="$LOG_DIR/vllm-${PORT}.log"
PID_FILE="$LOG_DIR/vllm-${PORT}.pid"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$LOG_DIR" "$(dirname "$MODEL_DIR")"

export HF_HOME
export HF_HUB_CACHE
export TRANSFORMERS_CACHE

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found in PATH." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Run this inside a GPU pod/container." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk 'NR>2 && $1 !~ /^#/ {print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating conda env: $ENV_NAME (python=$PYTHON_VERSION)"
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"

# Fix the libstdc++ / CXXABI mismatch you hit earlier.
conda install -y -c conda-forge "libstdcxx-ng>=14" "libgcc-ng>=14"

# Avoid inheriting a bad system library path.
unset LD_LIBRARY_PATH || true

python -m pip install --upgrade pip setuptools wheel
python -m pip install -U "huggingface_hub[cli]" requests openai
python -m pip install -U --extra-index-url https://wheels.vllm.ai/nightly vllm

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: hf CLI was not installed correctly." >&2
  exit 1
fi

# First sanity check: sqlite3 should import cleanly.
if ! python - <<'PY'
import sqlite3
print("sqlite3 import OK")
PY
then
  echo "sqlite3 import failed; forcing conda's libstdc++ via LD_PRELOAD ..."
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

  python - <<'PY'
import sqlite3
print("sqlite3 import OK with LD_PRELOAD")
PY
fi

echo "Hugging Face auth status (non-fatal if not logged in):"
hf auth whoami || true

echo "Storage paths:"
echo "  HF_HOME=$HF_HOME"
echo "  HF_HUB_CACHE=$HF_HUB_CACHE"
echo "  MODEL_DIR=$MODEL_DIR"
echo

echo "Downloading model to: $MODEL_DIR"
hf download "$MODEL_ID" --local-dir "$MODEL_DIR"

# Stop an older server on the same port if still running.
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${OLD_PID:-}" ]] && kill -0 "$OLD_PID" >/dev/null 2>&1; then
    echo "Stopping existing vLLM server on port $PORT (pid=$OLD_PID) ..."
    kill "$OLD_PID" || true
    sleep 2
  fi
  rm -f "$PID_FILE"
fi

echo
cat <<EOM
Starting vLLM with:
  model id: $MODEL_ID
  model dir: $MODEL_DIR
  port: $PORT
  max model len: $MAX_MODEL_LEN
  gpu memory utilization: $GPU_MEMORY_UTILIZATION

Notes:
- This script defaults to Python 3.10 to avoid the runtime issue you hit.
- It prefers /private for model/cache storage when available.
- Qwen3.5-4B can still be tight on a 12 GB MIG slice.
- If it OOMs, try MAX_MODEL_LEN=256 or switch to a smaller model.
- --language-model-only avoids loading the vision tower.
EOM

echo
nohup vllm serve "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "vLLM PID: $(cat "$PID_FILE")"
echo "Log file:  $LOG_FILE"

echo "Waiting for the API to come up ..."
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo
    echo "vLLM is up. Quick checks inside the pod:"
    echo "  curl http://127.0.0.1:${PORT}/v1/models"
    echo
    echo "On your LOCAL machine, open a tunnel with:"
    echo "  UCSD_USER=<your_ucsd_user> bash forward_vllm_port_local_fixed.sh"
    echo
    exit 0
  fi
  printf '.'
  sleep 2
done

echo
echo "vLLM did not become ready in time. Check the log with:" >&2
echo "  tail -n 200 $LOG_FILE" >&2
exit 1