#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-vllm-qwen35}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-2B}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-256}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
LOG_DIR="${LOG_DIR:-$HOME/vllm-logs}"

# Always use paths inside your writable home area.
PRIVATE_ROOT="$HOME/private"
DEFAULT_STORAGE_ROOT="$HOME"
if [[ -d "$PRIVATE_ROOT" ]]; then
  DEFAULT_STORAGE_ROOT="$PRIVATE_ROOT"
fi

# Ignore stale exported values from old sessions unless user explicitly passes them on this command line.
HF_HOME="${HF_HOME_OVERRIDE:-$DEFAULT_STORAGE_ROOT/hf_cache}"
HF_HUB_CACHE="${HF_HUB_CACHE_OVERRIDE:-$HF_HOME/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE_OVERRIDE:-$HF_HOME/transformers}"

# Use repo ID for serving; don't serve from local path.
MODEL_SOURCE="${MODEL_SOURCE:-$MODEL_ID}"

LOG_FILE="$LOG_DIR/vllm-${PORT}.log"
PID_FILE="$LOG_DIR/vllm-${PORT}.pid"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$LOG_DIR"

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

unset LD_PRELOAD || true
unset LD_LIBRARY_PATH || true

source /opt/conda/etc/profile.d/conda.sh

if ! conda env list | awk 'NR>2 && $1 !~ /^#/ {print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating conda env: $ENV_NAME (python=$PYTHON_VERSION)"
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -U "huggingface_hub[cli]" requests openai
python -m pip install -U --extra-index-url https://wheels.vllm.ai/nightly vllm
conda install -y -c conda-forge "libstdcxx-ng>=14" "libgcc-ng>=14" sqlite

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: hf CLI was not installed correctly." >&2
  exit 1
fi

python - <<'PY'
import sqlite3
print("sqlite3 import OK")
PY

echo "Hugging Face auth status (non-fatal if not logged in):"
hf auth whoami || true

echo "Storage paths:"
echo "  HF_HOME=$HF_HOME"
echo "  HF_HUB_CACHE=$HF_HUB_CACHE"
echo

# Optional warm download into cache.
echo "Caching model repo: $MODEL_ID"
hf download "$MODEL_ID" --cache-dir "$HF_HUB_CACHE" >/dev/null

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
  model source: $MODEL_SOURCE
  port: $PORT
  max model len: $MAX_MODEL_LEN
  max num seqs: $MAX_NUM_SEQS
  gpu memory utilization: $GPU_MEMORY_UTILIZATION

Notes:
- This serves from the Hugging Face repo ID, not a local model path.
- Cache is stored under: $HF_HUB_CACHE
- Qwen3.5-2B is a safer fit than 4B for your 12 GB MIG slice.
EOM

nohup env \
  HF_HOME="$HF_HOME" \
  HF_HUB_CACHE="$HF_HUB_CACHE" \
  TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
  LD_LIBRARY_PATH="$CONDA_PREFIX/lib" \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6" \
  "$CONDA_PREFIX/bin/vllm" serve "$MODEL_SOURCE" \
    --download-dir "$HF_HUB_CACHE" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "vLLM PID: $(cat "$PID_FILE")"
echo "Log file:  $LOG_FILE"

echo "Waiting for the API to come up ..."
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo
    echo "vLLM is up."
    echo "Test inside pod:"
    echo "  curl http://127.0.0.1:${PORT}/v1/models"
    exit 0
  fi
  printf '.'
  sleep 2
done

echo
echo "vLLM did not become ready in time. Check the log with:" >&2
echo "  tail -n 200 $LOG_FILE" >&2
exit 1