#!/bin/bash
# End-to-end GPUWatch validation on real GPU nodes. Run from node 0 as the
# ubuntu user after setup_node.sh succeeded on every node:
#
#   NODES="10.0.0.11 10.0.0.12" GPUS=4 /opt/gpuwatch/deploy/scripts/live_validation.sh
#
# Requires passwordless SSH from node 0 to every node (same user). Every number
# this prints is a REAL measurement on your hardware. Nothing is drained:
# the bring-up gate runs with --scheduler none.
# NOT VERIFIED IN CI - written for, but never executed on, real GPU hosts.
set -euo pipefail
: "${NODES:?space-separated private IPs, node 0 first}"
GPUS=${GPUS:-$(nvidia-smi -L | wc -l)}
REPO=${REPO:-/opt/gpuwatch}
PY=$REPO/.venv/bin/python
OUT=${OUT:-$REPO/reports/live-$(date -u +%Y%m%dT%H%M%SZ)}
PHASE_S=${PHASE_S:-300}
read -r -a IPS <<<"$NODES"
NNODES=${#IPS[@]}
MASTER=${IPS[0]}
mkdir -p "$OUT"
cd "$REPO"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/summary.txt"; }

log "1/6 bring-up gate (driver + DCGM diag level 3) on each node, dry-run"
for ip in "${IPS[@]}"; do
  # shellcheck disable=SC2029
  ssh "$ip" "cd $REPO && $PY -m bringup --backend live --node \$(hostname -s) --scheduler none \
      --checks driver,burn_in --expected-gpus $GPUS --report-dir $OUT/gate" \
    | tee -a "$OUT/gate.log" || log "gate FAILED on $ip (see $OUT/gate)"
done

log "2/6 NCCL baseline: single node, then across nodes"
all_reduce_perf -b 8M -e 1G -f 2 -g "$GPUS" | tee "$OUT/nccl_single.txt"
if [ "$NNODES" -gt 1 ] && command -v mpirun >/dev/null; then
  HOSTS=$(printf "%s:$GPUS," "${IPS[@]}"); HOSTS=${HOSTS%,}
  mpirun -np $((NNODES * GPUS)) -H "$HOSTS" -x LD_LIBRARY_PATH -x NCCL_DEBUG=WARN \
    all_reduce_perf -b 8M -e 1G -f 2 -g 1 | tee "$OUT/nccl_multi.txt" || log "multi-node nccl-tests failed"
fi
SINGLE=$(grep -oP 'Avg bus bandwidth\s*:\s*\K[\d.]+' "$OUT/nccl_single.txt" || echo 0)
log "measured single-node avg busbw: $SINGLE GB/s; suggested gate threshold (80%): \
$(awk -v b="$SINGLE" 'BEGIN{printf "%.1f", b*0.8}') GB/s"

log "3/6 Prometheus on node 0 scraping all exporters (10 s interval)"
TARGETS=$(printf '"%s:9400",' "${IPS[@]}"); TARGETS=${TARGETS%,}
cat > "$OUT/prometheus.yml" <<YAML
global: {scrape_interval: 10s, evaluation_interval: 10s}
rule_files: [/rules/*.yaml]
scrape_configs:
  - job_name: gpuwatch-exporter
    static_configs: [{targets: [$TARGETS]}]
YAML
docker rm -f gpuwatch-prom >/dev/null 2>&1 || true
docker run -d --name gpuwatch-prom --net host -v "$OUT/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  -v "$REPO/dashboards/alerts:/rules:ro" prom/prometheus:v2.55.1 >/dev/null
sleep 15

run_ddp() {  # $1 = label, rest = extra ddp_bench args
  local label=$1; shift
  local steps=100000
  for i in "${!IPS[@]}"; do
    # shellcheck disable=SC2029
    ssh "${IPS[$i]}" "cd $REPO && GPUWATCH_HOSTNAME=\$(hostname -s) timeout $PHASE_S \
      $REPO/.venv/bin/torchrun --nnodes $NNODES --nproc-per-node $GPUS --node-rank $i \
      --master-addr $MASTER --master-port 29500 deploy/workloads/ddp_bench.py --steps $steps \
      --textfile-dir /var/lib/gpuwatch/textfile $*" >"$OUT/ddp_${label}_node$i.log" 2>&1 &
  done
  wait || true
}

log "4/6 healthy DDP phase (${PHASE_S}s): expect NO detector events"
T0=$(date +%s); run_ddp healthy; T1=$(date +%s)
$PY -m detector --source prometheus --prom-url http://localhost:9090 --start "$T0" --end "$T1" \
  --interval 10 | tee "$OUT/events_healthy.jsonl"
log "healthy phase events: $(wc -l <"$OUT/events_healthy.jsonl")"

SLOW_RANK=$(( (NNODES - 1) * GPUS ))  # first GPU of the last node
log "5/6 injected straggler: rank $SLOW_RANK +30% GPU work (${PHASE_S}s): expect a straggler event on the last node"
T0=$(date +%s); run_ddp straggler --slow-rank "$SLOW_RANK" --slow-factor 1.3; T1=$(date +%s)
$PY -m detector --source prometheus --prom-url http://localhost:9090 --start "$T0" --end "$T1" \
  --interval 10 | tee "$OUT/events_straggler.jsonl"
log "straggler phase events: $(grep -c '"straggler"' "$OUT/events_straggler.jsonl" || true) straggler"

log "6/6 grounded AI summary of the first straggler event (needs ANTHROPIC_API_KEY)"
if [ -n "${ANTHROPIC_API_KEY:-}" ] && grep -q '"straggler"' "$OUT/events_straggler.jsonl"; then
  journalctl -k -o short-unix --since "-$((2 * PHASE_S))s" >"$OUT/kernel.log" 2>/dev/null || true
  $PY -m ai_diagnosis --backend anthropic --event "$OUT/events_straggler.jsonl" --kind straggler \
    --prom-url http://localhost:9090 --kernel-log "$OUT/kernel.log" | tee "$OUT/ai_diagnosis.md"
else
  log "skipped (no ANTHROPIC_API_KEY or no straggler event)"
fi
log "artifacts in $OUT"
