#!/usr/bin/env bash
set -euo pipefail

# Run this on your LOCAL machine.
# Example:
#   UCSD_USER=abc123 bash forward_vllm_port_local_fixed.sh
# Optional:
#   LOCAL_PORT=8000 REMOTE_PORT=8000 REMOTE_HOST=dsmlp-login.ucsd.edu \
#   REMOTE_BIND_HOST=128.54.65.160 bash forward_vllm_port_local_fixed.sh

UCSD_USER="${UCSD_USER:?Set UCSD_USER, e.g. UCSD_USER=abc123}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
REMOTE_PORT="${REMOTE_PORT:-8000}"
REMOTE_HOST="${REMOTE_HOST:-dsmlp-login.ucsd.edu}"
# Destination host resolved on the remote side of the SSH connection.
REMOTE_BIND_HOST="${REMOTE_BIND_HOST:-128.54.65.160}"

echo "Forwarding local http://127.0.0.1:${LOCAL_PORT} -> ${REMOTE_BIND_HOST}:${REMOTE_PORT} via ${UCSD_USER}@${REMOTE_HOST}"
exec ssh -N -L "${LOCAL_PORT}:${REMOTE_BIND_HOST}:${REMOTE_PORT}" "${UCSD_USER}@${REMOTE_HOST}"
