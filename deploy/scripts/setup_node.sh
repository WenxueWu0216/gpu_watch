#!/bin/bash
# Prepare one GPU node for GPUWatch live mode. Run as root on EACH node:
#   sudo /opt/gpuwatch/deploy/scripts/setup_node.sh
# Target: Ubuntu 22.04 with NVIDIA driver + Docker + NVIDIA container toolkit
# (e.g. AWS Deep Learning OSS Nvidia Driver AMI). Idempotent.
# NOT VERIFIED IN CI - written for, but never executed on, real GPU hosts.
set -euo pipefail
REPO=${REPO:-/opt/gpuwatch}
DCGM_EXPORTER_IMAGE=${DCGM_EXPORTER_IMAGE:-nvcr.io/nvidia/k8s/dcgm-exporter:3.3.8-3.6.0-ubuntu22.04}
TEXTFILE_DIR=/var/lib/gpuwatch/textfile
log() { echo "[setup_node] $*"; }

nvidia-smi -L || { echo "nvidia-smi failed: install/fix the NVIDIA driver first" >&2; exit 1; }

# 1. DCGM (dcgmi diag for burn-in). Comes from NVIDIA's CUDA apt repo.
if ! command -v dcgmi >/dev/null; then
  if ! apt-cache policy datacenter-gpu-manager | grep -q 'Candidate:.*[0-9]'; then
    log "adding NVIDIA CUDA apt repo"
    curl -fsSLo /tmp/cuda-keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i /tmp/cuda-keyring.deb
    apt-get update
  fi
  apt-get install -y datacenter-gpu-manager
fi
systemctl enable --now nvidia-dcgm 2>/dev/null || systemctl enable --now dcgm 2>/dev/null || \
  log "WARNING: could not start the DCGM host engine service; dcgmi diag will start an embedded one"

# 2. nccl-tests (single-node binary always; MPI build if an MPI is present).
if ! command -v all_reduce_perf >/dev/null; then
  apt-get install -y git build-essential
  if ! ldconfig -p | grep -q libnccl.so; then apt-get install -y libnccl2 libnccl-dev; fi
  rm -rf /opt/nccl-tests && git clone --depth 1 https://github.com/NVIDIA/nccl-tests /opt/nccl-tests
  MPI_HOME=""
  for d in /opt/amazon/openmpi /usr/lib/x86_64-linux-gnu/openmpi /opt/hpcx/ompi; do
    [ -x "$d/bin/mpicc" ] && MPI_HOME=$d && break
  done
  if [ -n "$MPI_HOME" ]; then
    make -C /opt/nccl-tests -j"$(nproc)" MPI=1 MPI_HOME="$MPI_HOME" CUDA_HOME=/usr/local/cuda
  else
    make -C /opt/nccl-tests -j"$(nproc)" CUDA_HOME=/usr/local/cuda
  fi
  ln -sf /opt/nccl-tests/build/all_reduce_perf /usr/local/bin/all_reduce_perf
fi

# 3. Python environment (detector, bring-up, ai_diagnosis).
apt-get install -y python3-venv
[ -d "$REPO/.venv" ] || python3 -m venv "$REPO/.venv"
"$REPO/.venv/bin/pip" install -q --upgrade pip
"$REPO/.venv/bin/pip" install -q -e "${REPO}[otel,llm]"

# 4. dcgm-exporter (localhost:9401) with GPUWatch's counter list.
mkdir -p "$TEXTFILE_DIR" && chmod 1777 "$TEXTFILE_DIR"
docker rm -f dcgm-exporter >/dev/null 2>&1 || true
docker run -d --restart unless-stopped --name dcgm-exporter --gpus all --cap-add SYS_ADMIN \
  -p 127.0.0.1:9401:9400 \
  -v "$REPO/deploy/dcgm/gpuwatch-counters.csv:/etc/dcgm-exporter/gpuwatch.csv:ro" \
  "$DCGM_EXPORTER_IMAGE" -f /etc/dcgm-exporter/gpuwatch.csv

# 5. GPUWatch exporter (built in a Go container; runs under systemd on :9400).
docker run --rm -v "$REPO/exporter:/src" -w /src golang:1.24 \
  sh -c 'CGO_ENABLED=0 go build -o bin/gpuwatch-exporter .'
install -m 0755 "$REPO/exporter/bin/gpuwatch-exporter" /usr/local/bin/gpuwatch-exporter
cat > /etc/systemd/system/gpuwatch-exporter.service <<UNIT
[Unit]
Description=GPUWatch exporter (live mode)
After=network-online.target docker.service

[Service]
ExecStart=/usr/local/bin/gpuwatch-exporter --mode=live --listen=:9400 \
  --dcgm-url=http://127.0.0.1:9401/metrics --textfile-dir=$TEXTFILE_DIR \
  --hostname=$(hostname -s) --force-hostname
Restart=always
DynamicUser=yes

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now gpuwatch-exporter
sleep 5
curl -fsS localhost:9400/metrics | grep -E '^(DCGM_FI_DEV_GPU_UTIL|gpuwatch_exporter_up)' | head -n 12
log "done. Exporter: http://$(hostname -I | awk '{print $1}'):9400/metrics"
