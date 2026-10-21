package main

import (
	"bytes"
	"context"
	"log/slog"
	"net/http"
	"sort"
	"sync"
	"time"

	dto "github.com/prometheus/client_model/go"
)

// Collector produces metric families on demand.
type Collector interface {
	Collect(ctx context.Context) (Families, error)
}

// Named pairs a collector with the source label used in self-metrics.
type Named struct {
	Name      string
	Collector Collector
}

// Handler serves /metrics by collecting all sources concurrently. A failing
// source is reported via gpuwatch_exporter_up{source="..."} 0 rather than
// failing the whole scrape.
type Handler struct {
	Sources []Named
	Timeout time.Duration
	Log     *slog.Logger
}

type result struct {
	name string
	fams Families
	err  error
	dur  time.Duration
}

// ServeHTTP implements http.Handler.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), h.Timeout)
	defer cancel()
	results := make([]result, len(h.Sources))
	var wg sync.WaitGroup
	for i, s := range h.Sources {
		wg.Add(1)
		go func(i int, s Named) {
			defer wg.Done()
			start := time.Now()
			fams, err := s.Collector.Collect(ctx)
			results[i] = result{s.Name, fams, err, time.Since(start)}
		}(i, s)
	}
	wg.Wait()
	sort.Slice(results, func(a, b int) bool { return results[a].name < results[b].name })
	all := Families{}
	for _, res := range results {
		up := 1.0
		if res.err != nil {
			up = 0
			if h.Log != nil {
				h.Log.Warn("collection failed", "source", res.name, "err", res.err)
			}
		}
		if res.fams != nil {
			all.merge(res.fams)
		}
		lbl := map[string]string{"source": res.name}
		all.add("gpuwatch_exporter_up", "1 if the source was collected successfully.", dto.MetricType_GAUGE, up, lbl)
		all.add("gpuwatch_exporter_collect_duration_seconds", "Time spent collecting the source.", dto.MetricType_GAUGE, res.dur.Seconds(), lbl)
	}
	var buf bytes.Buffer
	if err := all.write(&buf); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	_, _ = w.Write(buf.Bytes())
}
