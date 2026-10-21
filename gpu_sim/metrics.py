"""Metric names: the contract between simulator, exporter, detector and dashboards.

GPU metrics use the names that NVIDIA ``dcgm-exporter`` publishes, so the same
detector queries work against a real cluster. Job/network metrics that DCGM
does not provide are prefixed ``gpuwatch_`` and are produced by the DDP
benchmark workload (``deploy/workloads/ddp_bench.py``) via the textfile
directory of the Go exporter in live mode.

``exporter/metrics_contract.txt`` lists the same names; a test on each side
(Python and Go) keeps the three in sync.
"""

from __future__ import annotations

GPU_UTIL = "DCGM_FI_DEV_GPU_UTIL"
FB_USED = "DCGM_FI_DEV_FB_USED"
FB_FREE = "DCGM_FI_DEV_FB_FREE"
GPU_TEMP = "DCGM_FI_DEV_GPU_TEMP"
POWER_USAGE = "DCGM_FI_DEV_POWER_USAGE"
SM_CLOCK = "DCGM_FI_DEV_SM_CLOCK"
THROTTLE_REASONS = "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS"
ECC_SBE_TOTAL = "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL"
ECC_DBE_TOTAL = "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL"
XID_ERRORS = "DCGM_FI_DEV_XID_ERRORS"
NVLINK_BW = "DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL"

STEP_TIME = "gpuwatch_train_step_time_seconds"
NET_TX = "gpuwatch_net_tx_gbps"
NET_XMIT_WAIT = "gpuwatch_net_xmit_wait_rate"
NCCL_BUSBW = "gpuwatch_nccl_busbw_gb_per_second"

GPU_METRICS: tuple[str, ...] = (
    GPU_UTIL,
    FB_USED,
    FB_FREE,
    GPU_TEMP,
    POWER_USAGE,
    SM_CLOCK,
    THROTTLE_REASONS,
    ECC_SBE_TOTAL,
    ECC_DBE_TOTAL,
    XID_ERRORS,
    NVLINK_BW,
)
NODE_METRICS: tuple[str, ...] = (STEP_TIME, NET_TX, NET_XMIT_WAIT, NCCL_BUSBW)
ALL_METRICS: tuple[str, ...] = GPU_METRICS + NODE_METRICS

# Label names used by dcgm-exporter.
LABEL_NODE = "Hostname"
LABEL_GPU = "gpu"
