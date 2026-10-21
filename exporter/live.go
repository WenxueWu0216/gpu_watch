package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	dto "github.com/prometheus/client_model/go"
)

// DCGMSource scrapes an NVIDIA dcgm-exporter /metrics endpoint and passes
// the DCGM_* families through, adding Hostname if dcgm-exporter omitted it.
type DCGMSource struct {
	URL      string
	Hostname string
	// Force replaces any Hostname label (dcgm-exporter in a pod reports the pod name).
	Force  bool
	Client *http.Client
}

// Collect implements Collector.
func (d *DCGMSource) Collect(ctx context.Context) (Families, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, d.URL, nil)
	if err != nil {
		return nil, err
	}
	resp, err := d.Client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("dcgm-exporter returned HTTP %d", resp.StatusCode)
	}
	parsed, err := parseText(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("parse dcgm-exporter output: %w", err)
	}
	out := Families{}
	for name, mf := range parsed {
		if strings.HasPrefix(name, "DCGM_") {
			out[name] = mf
		}
	}
	if d.Hostname != "" {
		out.setLabel("Hostname", d.Hostname, d.Force)
	}
	return out, nil
}

// TextfileSource reads *.prom files written by workloads (e.g. the DDP
// benchmark's step time / busbw), like node_exporter's textfile collector.
type TextfileSource struct {
	Dir      string
	Hostname string
	Force    bool
	MaxAge   time.Duration
	Now      func() time.Time
}

// Collect implements Collector. Files older than MaxAge are ignored so a
// finished job does not keep reporting a stale step time.
func (t *TextfileSource) Collect(_ context.Context) (Families, error) {
	matches, err := filepath.Glob(filepath.Join(t.Dir, "*.prom"))
	if err != nil {
		return nil, err
	}
	now := time.Now
	if t.Now != nil {
		now = t.Now
	}
	out := Families{}
	var errs []error
	for _, path := range matches {
		st, err := os.Stat(path)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		if t.MaxAge > 0 && now().Sub(st.ModTime()) > t.MaxAge {
			continue
		}
		fh, err := os.Open(path)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		fams, err := parseText(fh)
		_ = fh.Close()
		if err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", path, err))
			continue
		}
		out.merge(fams)
	}
	if t.Hostname != "" {
		out.setLabel("Hostname", t.Hostname, t.Force)
	}
	return out, errors.Join(errs...)
}

// InfinibandSource computes per-node NIC throughput and transmit-wait rate
// from /sys/class/infiniband/<dev>/ports/<n>/counters. port_xmit_data is in
// units of 4 octets (IBTA spec); port_xmit_wait is in hardware ticks, so the
// exported rate is only meaningful relative to peer nodes.
type InfinibandSource struct {
	Root     string
	Hostname string
	Now      func() time.Time

	mu   sync.Mutex
	prev map[string]ibSample
}

type ibSample struct {
	at   time.Time
	data uint64
	wait uint64
}

func readCounter(path string) (uint64, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	return strconv.ParseUint(strings.TrimSpace(string(b)), 10, 64)
}

// Collect implements Collector. The first call only primes the counters.
func (s *InfinibandSource) Collect(_ context.Context) (Families, error) {
	ports, err := filepath.Glob(filepath.Join(s.Root, "*", "ports", "*", "counters"))
	if err != nil {
		return nil, err
	}
	now := time.Now
	if s.Now != nil {
		now = s.Now
	}
	ts := now()
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.prev == nil {
		s.prev = map[string]ibSample{}
	}
	var txGbps, waitRate float64
	have := false
	for _, dir := range ports {
		data, err1 := readCounter(filepath.Join(dir, "port_xmit_data"))
		wait, err2 := readCounter(filepath.Join(dir, "port_xmit_wait"))
		if err1 != nil || err2 != nil {
			continue
		}
		cur := ibSample{at: ts, data: data, wait: wait}
		prev, ok := s.prev[dir]
		s.prev[dir] = cur
		dt := cur.at.Sub(prev.at).Seconds()
		if !ok || dt <= 0 || cur.data < prev.data || cur.wait < prev.wait {
			continue // first sample or counter reset/wrap
		}
		txGbps += float64(cur.data-prev.data) * 4 * 8 / 1e9 / dt
		waitRate += float64(cur.wait-prev.wait) / dt
		have = true
	}
	out := Families{}
	if have {
		lbl := map[string]string{"Hostname": s.Hostname}
		out.add("gpuwatch_net_tx_gbps", "NIC transmit throughput (Gbit/s) from IB port_xmit_data.", dto.MetricType_GAUGE, txGbps, lbl)
		out.add("gpuwatch_net_xmit_wait_rate", "InfiniBand port_xmit_wait ticks per second.", dto.MetricType_GAUGE, waitRate, lbl)
	}
	return out, nil
}
