#!/bin/bash
# PyTorch DDP benchmark across nodes; exports per-rank step time and busbw
# for GPUWatch (NOT VERIFIED IN CI - requires a Slurm cluster with GPUs).
#
#   sbatch --nodes=2 --gpus-per-node=8 deploy/slurm/ddp_bench.sbatch
#   # inject a straggler on global rank 3 to test detection end to end:
#   EXTRA="--slow-rank 3 --slow-factor 1.3" sbatch deploy/slurm/ddp_bench.sbatch
#
#SBATCH --job-name=gpuwatch-ddp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}
REPO=${REPO:-$SLURM_SUBMIT_DIR}
TEXTFILE_DIR=${TEXTFILE_DIR:-/var/lib/gpuwatch/textfile}
STEPS=${STEPS:-2000}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR MASTER_PORT=${MASTER_PORT:-29500}
# Uncomment to see which NCCL transport is used (IB vs socket fallback):
# export NCCL_DEBUG=INFO

# One torchrun launcher per node; torchrun spawns one process per GPU.
# GPUWATCH_HOSTNAME must match the exporter's Hostname label (node name).
srun --kill-on-bad-exit=1 bash -c "GPUWATCH_HOSTNAME=\$(hostname -s) torchrun \
  --nnodes=$SLURM_NNODES --nproc-per-node=$GPUS_PER_NODE \
  --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv-id=$SLURM_JOB_ID \
  $REPO/deploy/workloads/ddp_bench.py --steps $STEPS --textfile-dir $TEXTFILE_DIR ${EXTRA:-}"
