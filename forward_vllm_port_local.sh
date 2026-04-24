#!/usr/bin/env bash
set -euo pipefail

# Run this on your LOCAL machine, not on UCSD.
# Example:
#   UCSD_USER=abc123 REMOTE_PORT=8000 LOCAL_PORT=8000 bash forward_vllm_port_local.sh

UCSD_USER="${UCSD_USER:?Set UCSD_USER, e.g. UCSD_USER=abc123}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
REMOTE_PORT="${REMOTE_PORT:-8000}"
REMOTE_HOST="${REMOTE_HOST:-dsmlp-login.ucsd.edu}"
REMOTE_IP="${REMOTE_IP:-128.54.65.160}"

exec ssh -N -L "${LOCAL_PORT}:${REMOTE_IP}:${REMOTE_PORT}" "${UCSD_USER}@${REMOTE_HOST}"
