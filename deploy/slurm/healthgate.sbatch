#!/bin/bash
# Full GPUWatch bring-up gate on one node, run as an exclusive job
# (NOT VERIFIED IN CI). Draining requires Slurm operator rights, so by default
# this only REPORTS (dry-run); add --execute via GATE_ARGS when run as an operator.
#
#   sbatch -w gpu-node-01 --export=ALL,MIN_BUSBW=350 deploy/slurm/healthgate.sbatch
#
#SBATCH --job-name=gpuwatch-gate
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --time=00:45:00
#SBATCH --output=%x-%N-%j.out
set -euo pipefail
REPO=${REPO:-$SLURM_SUBMIT_DIR}
NODE=$(hostname -s)
: "${MIN_BUSBW:?set MIN_BUSBW (GB/s) for this GPU model/interconnect}"
# GATE_ARGS is intentionally word-split (e.g. "--execute --burn-in gpu_burn").
# shellcheck disable=SC2086
"${PYTHON:-python3}" -m bringup --backend live --node "$NODE" --scheduler slurm \
  --expected-gpus "${SLURM_GPUS_ON_NODE:-8}" --min-busbw "$MIN_BUSBW" \
  --nccl-cmd "all_reduce_perf -b 8M -e 1G -f 2 -g ${SLURM_GPUS_ON_NODE:-8}" \
  --report-dir "${REPORT_DIR:-$REPO/reports}" ${GATE_ARGS:-}
