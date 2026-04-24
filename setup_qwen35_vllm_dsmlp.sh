#!/usr/bin/env bash
set -euo pipefail

# Run this INSIDE your GPU pod/container, not on dsmlp-login.
# Recommended from dsmlp-login before entering the pod:
#   IDENTITY_PROXY_PORTS=1 launch.sh -g 1 -b
#   kubectl get pod
#   kubesh <your-pod-name>
# Then inside the pod:
#   bash setup_qwen35_vllm_dsmlp_fixed.sh
#
# Optional overrides:
#   ENV_NAME=vllm-qwen35 PYTHON_VERSION=3.12 MODEL_ID=Qwen/Qwen3.5-4B \
#   MODEL_DIR=$HOME/models/Qwen3.5-4B PORT=8000 MAX_MODEL_LEN=1024 \
#   GPU_MEMORY_UTILIZATION=0.80 bash setup_qwen35_vllm_dsmlp_fixed.sh

ENV_NAME="${ENV_NAME:-vllm-qwen35}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-4B}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/$(basename "$MODEL_ID")}" 
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
LOG_DIR="${LOG_DIR:-$HOME/vllm-logs}"
LOG_FILE="$LOG_DIR/vllm-${PORT}.log"
PID_FILE="$LOG_DIR/vllm-${PORT}.pid"

mkdir -p "$HF_HOME" "$LOG_DIR" "$(dirname "$MODEL_DIR")"
export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

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

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -U huggingface_hub requests openai
# vLLM wheels are provided via the nightly index used by Qwen docs.
python -m pip install -U --extra-index-url https://wheels.vllm.ai/nightly vllm

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: hf CLI was not installed correctly." >&2
  exit 1
fi

echo "Hugging Face auth status (non-fatal if not logged in):"
hf auth whoami || true

echo "Downloading model to: $MODEL_DIR"
# For a public repo this works without login. --resume-download is no longer needed.
hf download "$MODEL_ID" --local-dir "$MODEL_DIR"

# Stop an older server on the same port if it is still running.
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" >/dev/null 2>&1; then
  echo "Stopping existing vLLM server on port $PORT ..."
  kill "$(cat "$PID_FILE")" || true
  sleep 2
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
- Qwen3.5-4B can still be tight on a 12 GB MIG slice.
- If it OOMs, try MAX_MODEL_LEN=512 first.
- --language-model-only avoids loading the vision tower.
EOM

echo
nohup vllm serve "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --language-model-only \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "vLLM PID: $(cat "$PID_FILE")"
echo "Log file:  $LOG_FILE"

echo "Waiting for the API to come up ..."
for _ in $(seq 1 120); do
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
