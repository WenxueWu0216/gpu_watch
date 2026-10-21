package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"

	dto "github.com/prometheus/client_model/go"
)

// simGPU mirrors gpu_sim.model.GPUSample.
type simGPU struct {
	Node            string  `json:"node"`
	GPU             int     `json:"gpu"`
	UtilPct         float64 `json:"util_pct"`
	MemUsedMiB      float64 `json:"mem_used_mib"`
	MemTotalMiB     float64 `json:"mem_total_mib"`
	TempC           float64 `json:"temp_c"`
	PowerW          float64 `json:"power_w"`
	SMClockMHz      float64 `json:"sm_clock_mhz"`
	ThrottleReasons int64   `json:"throttle_reasons"`
	EccSBETotal     int64   `json:"ecc_sbe_total"`
	EccDBETotal     int64   `json:"ecc_dbe_total"`
	Xid             int64   `json:"xid"`
	NVLinkGbps      float64 `json:"nvlink_gbps"`
	Available       bool    `json:"available"`
}

// simNode mirrors gpu_sim.model.NodeSample. Nullable fields are pointers.
type simNode struct {
	Node         string   `json:"node"`
	StepTimeS    *float64 `json:"step_time_s"`
	NetTxGbps    float64  `json:"net_tx_gbps"`
	NetXmitWait  float64  `json:"net_xmit_wait"`
	NCCLBusbwGBs *float64 `json:"nccl_busbw_gb_s"`
}

// SimSnapshot mirrors gpu_sim.model.Snapshot (logs are not exported as metrics).
type SimSnapshot struct {
	TS    float64   `json:"ts"`
	GPUs  []simGPU  `json:"gpus"`
	Nodes []simNode `json:"nodes"`
}

// SimSource fetches snapshots from the Python simulator's HTTP server.
type SimSource struct {
	URL    string
	Client *http.Client
}

// Fetch retrieves and decodes the latest snapshot.
func (s *SimSource) Fetch(ctx context.Context) (*SimSnapshot, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.URL, nil)
	if err != nil {
		return nil, err
	}
	resp, err := s.Client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("sim returned HTTP %d", resp.StatusCode)
	}
	var snap SimSnapshot
	if err := json.NewDecoder(resp.Body).Decode(&snap); err != nil {
		return nil, fmt.Errorf("decode snapshot: %w", err)
	}
	return &snap, nil
}

// Collect implements Collector.
func (s *SimSource) Collect(ctx context.Context) (Families, error) {
	snap, err := s.Fetch(ctx)
	if err != nil {
		return nil, err
	}
	return SnapshotFamilies(snap), nil
}

// SnapshotFamilies converts a simulator snapshot into DCGM-named metric families.
func SnapshotFamilies(snap *SimSnapshot) Families {
	f := Families{}
	gauge, counter := dto.MetricType_GAUGE, dto.MetricType_COUNTER
	for _, g := range snap.GPUs {
		lbl := map[string]string{
			"Hostname":  g.Node,
			"gpu":       strconv.Itoa(g.GPU),
			"UUID":      fmt.Sprintf("GPU-sim-%s-%d", g.Node, g.GPU),
			"device":    fmt.Sprintf("nvidia%d", g.GPU),
			"modelName": "NVIDIA H100 80GB HBM3 (simulated)",
		}
		// The XID field comes from driver events and is still reported when
		// the GPU itself is unreadable (fallen off the bus).
		f.add("DCGM_FI_DEV_XID_ERRORS", "Value of the last XID error encountered.", gauge, float64(g.Xid), lbl)
		f.add("DCGM_FI_DEV_ECC_SBE_VOL_TOTAL", "Total number of single-bit volatile ECC errors.", counter, float64(g.EccSBETotal), lbl)
		f.add("DCGM_FI_DEV_ECC_DBE_VOL_TOTAL", "Total number of double-bit volatile ECC errors.", counter, float64(g.EccDBETotal), lbl)
		if !g.Available {
			continue
		}
		f.add("DCGM_FI_DEV_GPU_UTIL", "GPU utilization (in %).", gauge, g.UtilPct, lbl)
		f.add("DCGM_FI_DEV_FB_USED", "Framebuffer memory used (in MiB).", gauge, g.MemUsedMiB, lbl)
		f.add("DCGM_FI_DEV_FB_FREE", "Framebuffer memory free (in MiB).", gauge, g.MemTotalMiB-g.MemUsedMiB, lbl)
		f.add("DCGM_FI_DEV_GPU_TEMP", "GPU temperature (in C).", gauge, g.TempC, lbl)
		f.add("DCGM_FI_DEV_POWER_USAGE", "Power draw (in W).", gauge, g.PowerW, lbl)
		f.add("DCGM_FI_DEV_SM_CLOCK", "SM clock frequency (in MHz).", gauge, g.SMClockMHz, lbl)
		f.add("DCGM_FI_DEV_CLOCK_THROTTLE_REASONS", "Throttling reasons bitmask.", gauge, float64(g.ThrottleReasons), lbl)
		f.add("DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL", "Total NVLink bandwidth (simulated, GB/s).", gauge, g.NVLinkGbps, lbl)
	}
	for _, n := range snap.Nodes {
		lbl := map[string]string{"Hostname": n.Node}
		if n.StepTimeS != nil {
			f.add("gpuwatch_train_step_time_seconds", "Rank-local time-to-barrier of the last training steps.", gauge, *n.StepTimeS, lbl)
		}
		f.add("gpuwatch_net_tx_gbps", "NIC transmit throughput (Gbit/s).", gauge, n.NetTxGbps, lbl)
		f.add("gpuwatch_net_xmit_wait_rate", "InfiniBand port_xmit_wait ticks per second.", gauge, n.NetXmitWait, lbl)
		if n.NCCLBusbwGBs != nil {
			f.add("gpuwatch_nccl_busbw_gb_per_second", "All-reduce bus bandwidth (GB/s, nccl-tests definition).", gauge, *n.NCCLBusbwGBs, lbl)
		}
	}
	f.add("gpuwatch_sim_time_seconds", "Simulated time of the snapshot.", gauge, snap.TS, map[string]string{})
	return f
}
