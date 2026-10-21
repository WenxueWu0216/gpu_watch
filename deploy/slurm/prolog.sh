#!/bin/bash
# Slurm Prolog hook (slurm.conf: Prolog=/etc/slurm/prolog.d/gpuwatch.sh,
# PrologFlags=Alloc). Runs as root on each allocated node before the job.
# A non-zero exit makes slurmd DRAIN the node and requeue the job, so this
# runs only the fast driver/inventory check; burn-in and NCCL belong in
# healthgate.sbatch. (NOT VERIFIED IN CI.)
set -uo pipefail
GPUWATCH_PY=${GPUWATCH_PY:-/opt/gpuwatch/.venv/bin/python}
EXPECTED=${GPUWATCH_EXPECTED_GPUS:-8}
REPORT_DIR=${GPUWATCH_REPORT_DIR:-/var/lib/gpuwatch/reports}
cd /opt/gpuwatch || exit 0   # not installed: do not block jobs
timeout 120 "$GPUWATCH_PY" -m bringup --backend live --node "$(hostname -s)" \
  --checks driver --expected-gpus "$EXPECTED" --scheduler none --report-dir "$REPORT_DIR" \
  >>/var/log/gpuwatch-prolog.log 2>&1
rc=$?
if [ "$rc" -eq 2 ]; then
  echo "gpuwatch: driver/GPU check failed on $(hostname -s); see $REPORT_DIR" >&2
  exit 1
fi
exit 0   # tool errors (rc 1, timeouts) must not drain healthy nodes
