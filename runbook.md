# GPUWatch runbook: top 5 GPU cluster failures

Each section lists the **detection signal** (GPUWatch detector event kind, the
backstop Prometheus alert, and the raw metrics), **diagnosis steps** to confirm
the cause, the **fix**, and how to **verify** before the node rejoins the pool.

General rules:

- **Take the node out of scheduling first, debug second.** Use the bring-up gate's
  adapters (`python -m bringup ... --execute`) or directly:
  `scontrol update NodeName=<node> State=DRAIN Reason="gpuwatch: <why>"` /
  `kubectl cordon <node> && kubectl drain <node> --ignore-daemonsets --delete-emptydir-data`.
- **Collect evidence before resetting anything**: `nvidia-bug-report.sh` (writes
  `nvidia-bug-report.log.gz`), `dmesg -T > dmesg.txt`, `nvidia-smi -q > smi.txt`.
  A reboot/reset clears volatile ECC counters and the kernel ring buffer.
- **A node rejoins only after the bring-up gate passes**:
  `python -m bringup --backend live --node <node> --scheduler slurm --min-busbw <GB/s> --execute`.
- `python -m ai_diagnosis` can summarize an alert with cited evidence; treat its output as a
  starting hypothesis, not a verdict.

---

## 1. XID errors

**Detection signal**

- Detector: `xid_error` (critical for hardware XIDs, warning for application-typical ones),
  often together with `gpu_missing` when the GPU stops reporting.
- Prometheus: `GPUXidCritical`, `GPUXidOther`.
- Metric: `DCGM_FI_DEV_XID_ERRORS` (value = last XID). Kernel log: `NVRM: Xid (PCI:...): <code>, ...`.

**Common codes** (see NVIDIA's "XID Errors" documentation for the full list):

| XID | Meaning | Usual owner |
|---|---|---|
| 13, 31, 43, 45 | graphics exception / MMU fault / GPU stopped processing / preemptive cleanup | usually the application (bad kernel, OOB access); hardware only if it repeats across jobs on one GPU |
| 48 | double-bit (uncorrectable) ECC error | hardware - see section 2 |
| 63, 64 | row-remapping event recorded / row-remapping failure (A100 and newer) | hardware - see section 2 |
| 74 | NVLink error | hardware/interconnect - see section 5 |
| 79 | GPU has fallen off the bus | hardware/platform (PCIe, power, thermal) |
| 92 | high single-bit ECC error rate | hardware - see section 2 |
| 94, 95 | contained / uncontained ECC error | hardware - see section 2 |
| 119, 120 | GSP firmware RPC timeout / GSP error | driver/firmware |

**Diagnosis**

1. `dmesg -T | grep -i -E "xid|nvrm"` - code, PCI bus id, and whether it repeats.
2. Map bus id to GPU index: `nvidia-smi --query-gpu=index,pci.bus_id,serial --format=csv`.
3. For XID 79: look for PCIe errors around the same time:
   `journalctl -k | grep -i -E "aer|pcie bus error"` and
   `lspci -vvv -s <bus_id> | grep -E "LnkSta|LnkCap"` (link speed/width degraded?).
   Check GPU temperature and power history in the *Node GPUs* dashboard just before the event.
4. For 13/31/43: check whether the same job fails on other nodes (application bug) or
   different jobs fail on this GPU (hardware).
5. For 119/120: note the driver version; check NVIDIA release notes for GSP fixes.

**Fix**

- Application XIDs: hand back to the job owner with the failing kernel/rank; no node action
  unless repeated on the same GPU.
- XID 79: full power cycle (not a warm reboot) via BMC: `ipmitool chassis power cycle`.
  If it recurs on the same slot: reseat GPU/riser, check PSU, then RMA.
- XID 119/120: upgrade driver/firmware; as a workaround some sites disable GSP firmware
  (driver-version dependent - check the release notes).

**Verify**: bring-up gate passes (driver check finds all GPUs; DCGM diag level 3 passes);
no new XIDs in `dmesg` after a 1-hour burn-in.

---

## 2. ECC errors

**Detection signal**

- Detector: `ecc_error` - critical on any increase of uncorrectable (DBE) errors since the
  detector started (latched until the volatile counter resets), warning on >= 10 correctable
  (SBE) errors within one 6-scrape window.
- Prometheus: `GPUEccUncorrectable`, `GPUEccCorrectableBurst`.
- Metrics: `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL`, `DCGM_FI_DEV_ECC_SBE_VOL_TOTAL`; XID 48/63/64/92/94/95.

**Diagnosis**

1. `nvidia-smi -q -d ECC` - volatile vs aggregate, SRAM vs DRAM counts.
2. A100/H100: `nvidia-smi -q -d ROW_REMAPPER` -
   *Pending: Yes* means a remap is queued and needs a GPU reset;
   *Remapping Failure Occurred: Yes* means spare rows are exhausted.
   Older GPUs (V100): `nvidia-smi -q -d PAGE_RETIREMENT`.
3. Check whether a DBE killed a job (XID 48/94/95 in `dmesg`, CUDA "uncorrectable ECC error"
   in the job log).

**Fix**

- Remap pending: drain, then reset the GPU (`nvidia-smi -r -i <idx>`, requires no processes on
  it) or reboot, so the remap takes effect.
- Remapping failure, or DBEs recurring after remap: RMA the GPU.
- Isolated SBE bursts with no DBE and no pending remap: keep in service, but watch the trend on
  the *Node GPUs* dashboard; repeated bursts are an early RMA indicator.

**Verify**: `nvidia-smi -q -d ROW_REMAPPER` shows no pending remaps; bring-up gate passes
(driver check reports 0 uncorrectable errors; DCGM diag memory tests pass).

---

## 3. Thermal throttle

**Detection signal**

- Detector: `thermal_throttle` - SW/HW thermal throttle bit set (or temperature >= 87 C with
  an SM clock >= 10% below same-node peers, not power-capped) for 3 consecutive scrapes;
  critical with `HW_THERMAL` or >= 92 C. Usually followed by a `straggler` event on the same
  node, whose evidence names the throttled GPU as the likely cause.
- Prometheus: `GPUThermalThrottle`, `GPUTemperatureCritical`.
- Metrics: `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` (0x20 SW thermal, 0x40 HW thermal; 0x4 is
  power cap, which is *not* a thermal problem), `DCGM_FI_DEV_GPU_TEMP`, `DCGM_FI_DEV_SM_CLOCK`.

**Diagnosis**

1. `nvidia-smi -q -d PERFORMANCE` - the clock throttle / "clocks event" reasons that are active.
2. `nvidia-smi -q -d TEMPERATURE` - current vs slowdown/shutdown temperature.
3. One hot GPU with cool peers -> local cause (fan, heatsink, airflow obstruction).
   All GPUs on the node hot -> chassis fans or inlet temperature:
   `ipmitool sdr type Fan`, `ipmitool sdr type Temperature`.
   Several nodes in a rack hot -> facility/cooling issue; escalate to datacenter ops.
4. Confirm the impact: the *Cluster overview* step-time ratio for that node.

**Fix**

- Fans/BMC fault: replace fan module; reset BMC fan policy.
- Airflow: remove obstructions/blanking-panel gaps; check hot/cold aisle containment.
- Persisting on one GPU: reseat heatsink / re-paste or RMA.
- Short-term mitigation while waiting for repair: drain the node (a throttled GPU slows every
  rank of a synchronous job), or lower the power limit (`nvidia-smi -pl <W> -i <idx>`) if the
  node must stay in service.

**Verify**: DCGM diag level 3 passes without thermal violations; during the gate's burn-in the
throttle bits stay clear and temperature stays below the slowdown threshold.

---

## 4. Straggler node

**Detection signal**

- Detector: `straggler` - the node's windowed (12-scrape) median rank time-to-barrier is a
  robust outlier vs the cluster median (modified z > 3.5) **and** >= 5% slower, for 3
  consecutive evaluations; critical at >= 25% slower. Evidence includes a pattern hint:
  *GPU-bound* (node's GPU utilization is also an outlier upward: GPUs are doing more work per
  step, e.g. lower clocks) or *host/input-bound* (utilization unchanged: GPUs wait on the host).
- Prometheus: `NodeStraggler` (step time > 110% of median for 2 minutes).
- Metrics: `gpuwatch_train_step_time_seconds` (per rank, from the workload),
  `DCGM_FI_DEV_GPU_UTIL`.

Note: in synchronous data-parallel training *every* rank's wall-clock step time is the same;
only the per-rank time-to-barrier (compute + input pipeline, excluding the wait in
all-reduce) reveals which node is slow. The DDP benchmark in `deploy/workloads/` exports it.

**Diagnosis**

1. Correlated alerts on the same node first (`thermal_throttle`, `ecc_error`): if present,
   follow that section.
2. GPU-bound: `nvidia-smi -q -d CLOCK,PERFORMANCE` (clocks lower than peers?),
   `nvidia-smi -q -d PCI` (link generation/width below max?), `nvidia-smi topo -m`.
3. Host/input-bound: CPU saturation or steal (`mpstat -P ALL 1`, `top`), data-loader worker
   count, storage latency (`iostat -x 1`), page cache pressure, CPU/NUMA affinity of the ranks
   (`numactl --show` inside the job), noisy neighbours / stray processes (`nvidia-smi` process
   list, `ps aux --sort=-%cpu`).
4. Live stack sample of the slow rank: `py-spy dump --pid <pid>`.

**Fix**

- Hardware cause (clocks, PCIe width): drain and repair; re-gate.
- Host cause: fix affinity / data-loader settings, kill stray processes, move data to local
  NVMe; or exclude the node for the current job (`sbatch --exclude=<node>`).

**Verify**: after the fix, the node's step-time ratio returns to ~1.0 in the *Cluster overview*
dashboard and the detector emits `resolved: straggler`.

---

## 5. NCCL slow link

**Detection signal**

- Detector: `nccl_slow_link` - the node's NIC transmit-wait rate (InfiniBand
  `port_xmit_wait`, summed over ports) is a robust outlier (z > 3.5) and >= 30% above the
  cluster median for 3 evaluations; critical at >= 3x. Usually accompanied by the
  cluster-level `nccl_busbw_degraded` (all-reduce bus bandwidth >= 25% below its pre-incident
  baseline), because one slow link in a ring slows every rank.
- Prometheus: `NCCLSlowLink`, `NCCLBusBandwidthDrop`.
- Metrics: `gpuwatch_net_xmit_wait_rate`, `gpuwatch_net_tx_gbps`,
  `gpuwatch_nccl_busbw_gb_per_second`. The bring-up gate's `nccl_allreduce` check catches the
  same problem before a node joins.

**Diagnosis**

1. Rerun the job (or nccl-tests) with `NCCL_DEBUG=INFO`: confirm the transport is `NET/IB`
   (or the expected plugin, e.g. AWS OFI for EFA) and not a fallback to `NET/Socket`; check
   GPUDirect RDMA is in use (`nvidia-peermem` loaded).
2. Link state and speed: `ibstat` / `ibv_devinfo` (State Active, expected Rate),
   `mlxlink -d <dev> -m` (cable/transceiver, BER).
3. Error counters: `perfquery -x` (SymbolErrorCounter, LinkDownedCounter,
   PortRcvErrors, PortXmitWait). Rising symbol errors -> bad cable/transceiver.
4. Intra-node: `nvidia-smi nvlink -s` (link speeds) and `nvidia-smi nvlink -e` (error
   counters); XID 74 indicates NVLink errors.
5. Bisect with nccl-tests between pairs of nodes (`all_reduce_perf -b 1G -e 1G -g 1` across
   two hosts) to find the slow pair.
6. RoCE: check PFC/ECN configuration and MTU on host and switch.

**Fix**: reseat/replace the cable or transceiver; move to another switch port; fix MTU/PFC
configuration; for NVLink errors, drain and RMA. Re-gate with
`--nccl-cmd "srun -N2 --ntasks-per-node=<gpus> all_reduce_perf -b 8M -e 1G -f 2 -g 1"`.

**Verify**: nccl-tests average bus bandwidth back above the gate threshold; transmit-wait
ratio back to ~1x the cluster median; `nccl_busbw_degraded` resolves.
