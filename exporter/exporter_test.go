package main

import (
	"bufio"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func readFile(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func serveBytes(t *testing.T, body []byte, status int) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(status)
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func scrape(t *testing.T, h http.Handler) string {
	t.Helper()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d", rec.Code)
	}
	return rec.Body.String()
}

func contract(t *testing.T) []string {
	t.Helper()
	fh, err := os.Open("metrics_contract.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = fh.Close() }()
	var names []string
	sc := bufio.NewScanner(fh)
	for sc.Scan() {
		if s := strings.TrimSpace(sc.Text()); s != "" {
			names = append(names, s)
		}
	}
	return names
}

func TestSimModeExportsEveryContractMetric(t *testing.T) {
	srv := serveBytes(t, readFile(t, "../tests/fixtures/snapshot_healthy.json"), 200)
	h := &Handler{Sources: BuildCollectors(Config{Mode: "sim", SimURL: srv.URL, Timeout: time.Second}), Timeout: time.Second}
	out := scrape(t, h)
	for _, name := range contract(t) {
		if !strings.Contains(out, "\n"+name+"{") {
			t.Errorf("metric %s missing from sim output", name)
		}
	}
	if !strings.Contains(out, `gpuwatch_exporter_up{source="sim"} 1`) {
		t.Error("expected up=1")
	}
	if !strings.Contains(out, `DCGM_FI_DEV_GPU_UTIL{Hostname="node-00",UUID="GPU-sim-node-00-0",device="nvidia0",gpu="0",modelName="NVIDIA H100 80GB HBM3 (simulated)"} 85.97`) {
		t.Errorf("util sample not rendered as expected:\n%s", out)
	}
	if !strings.Contains(out, "# TYPE DCGM_FI_DEV_ECC_DBE_VOL_TOTAL counter") {
		t.Error("ECC counters must be typed counter")
	}
	// FB_FREE is derived as total - used.
	if !strings.Contains(out, `DCGM_FI_DEV_FB_FREE{Hostname="node-00",UUID="GPU-sim-node-00-0"`) {
		t.Error("FB_FREE missing")
	}
}

func TestSimModeOffBusGPUOnlyReportsXidAndECC(t *testing.T) {
	var snap SimSnapshot
	srv := serveBytes(t, readFile(t, "../tests/fixtures/snapshot_xid79.json"), 200)
	src := &SimSource{URL: srv.URL, Client: http.DefaultClient}
	got, err := src.Fetch(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	snap = *got
	f := SnapshotFamilies(&snap)
	var buf strings.Builder
	if err := f.write(&buf); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	if !strings.Contains(out, `DCGM_FI_DEV_XID_ERRORS{Hostname="node-01",UUID="GPU-sim-node-01-1",device="nvidia1",gpu="1",modelName="NVIDIA H100 80GB HBM3 (simulated)"} 79`) {
		t.Errorf("xid 79 missing:\n%s", out)
	}
	if strings.Contains(out, `DCGM_FI_DEV_GPU_UTIL{Hostname="node-01",UUID="GPU-sim-node-01-1"`) {
		t.Error("off-bus GPU must not report utilization")
	}
	// Hung job: step time and busbw are null and must be absent.
	if strings.Contains(out, "gpuwatch_train_step_time_seconds{") || strings.Contains(out, "gpuwatch_nccl_busbw_gb_per_second{") {
		t.Error("null node metrics must be omitted")
	}
}

func TestSimSourceDownReportsUpZero(t *testing.T) {
	srv := serveBytes(t, []byte("boom"), 500)
	h := &Handler{Sources: BuildCollectors(Config{Mode: "sim", SimURL: srv.URL, Timeout: time.Second}), Timeout: time.Second}
	out := scrape(t, h)
	if !strings.Contains(out, `gpuwatch_exporter_up{source="sim"} 0`) {
		t.Errorf("expected up=0:\n%s", out)
	}
	if strings.Contains(out, "DCGM_FI_DEV_GPU_UTIL") {
		t.Error("no data expected when source is down")
	}
}

func TestSimSourceBadJSON(t *testing.T) {
	srv := serveBytes(t, []byte("{not json"), 200)
	_, err := (&SimSource{URL: srv.URL, Client: http.DefaultClient}).Collect(context.Background())
	if err == nil {
		t.Fatal("expected decode error")
	}
}

func TestLiveModeMergesDCGMTextfileAndIB(t *testing.T) {
	dcgm := serveBytes(t, readFile(t, "testdata/dcgm_exporter_sample.txt"), 200)
	textDir := t.TempDir()
	prom := "# TYPE gpuwatch_train_step_time_seconds gauge\n" +
		"gpuwatch_train_step_time_seconds{rank=\"0\"} 0.412\n" +
		"gpuwatch_train_step_time_seconds{rank=\"1\"} 0.415\n"
	if err := os.WriteFile(filepath.Join(textDir, "ddp.prom"), []byte(prom), 0o644); err != nil {
		t.Fatal(err)
	}
	stale := filepath.Join(textDir, "old.prom")
	if err := os.WriteFile(stale, []byte("gpuwatch_nccl_busbw_gb_per_second 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-time.Hour)
	if err := os.Chtimes(stale, old, old); err != nil {
		t.Fatal(err)
	}
	ib := t.TempDir()
	counters := filepath.Join(ib, "mlx5_0", "ports", "1", "counters")
	if err := os.MkdirAll(counters, 0o755); err != nil {
		t.Fatal(err)
	}
	write := func(data, wait string) {
		_ = os.WriteFile(filepath.Join(counters, "port_xmit_data"), []byte(data+"\n"), 0o644)
		_ = os.WriteFile(filepath.Join(counters, "port_xmit_wait"), []byte(wait+"\n"), 0o644)
	}
	cfg := Config{Mode: "live", DCGMURL: dcgm.URL, TextfileDir: textDir, IBRoot: ib, Hostname: "gpu-a", Timeout: time.Second}
	srcs := BuildCollectors(cfg)
	if len(srcs) != 3 {
		t.Fatalf("expected 3 live sources, got %d", len(srcs))
	}
	clock := time.Unix(1000, 0)
	srcs[2].Collector.(*InfinibandSource).Now = func() time.Time { return clock }
	h := &Handler{Sources: srcs, Timeout: time.Second}

	write("0", "0")
	first := scrape(t, h)
	if strings.Contains(first, "gpuwatch_net_tx_gbps{") {
		t.Error("first IB scrape only primes counters")
	}
	// 10 s later: 1.25e9 words * 4 B * 8 bit / 1e9 / 10 s = 4 Gbit/s; 5000 ticks / 10 s = 500/s.
	clock = clock.Add(10 * time.Second)
	write("1250000000", "5000")
	out := scrape(t, h)
	for _, want := range []string{
		`DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-aaaa",device="nvidia0",modelName="NVIDIA H100 80GB HBM3",Hostname="gpu-a"} 97`,
		`DCGM_FI_DEV_GPU_UTIL{gpu="1",UUID="GPU-bbbb",device="nvidia1",modelName="NVIDIA H100 80GB HBM3",Hostname="gpu-a"} 95`,
		`gpuwatch_train_step_time_seconds{rank="1",Hostname="gpu-a"} 0.415`,
		`gpuwatch_net_tx_gbps{Hostname="gpu-a"} 4`,
		`gpuwatch_net_xmit_wait_rate{Hostname="gpu-a"} 500`,
		`gpuwatch_exporter_up{source="dcgm"} 1`,
		`gpuwatch_exporter_up{source="textfile"} 1`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in:\n%s", want, out)
		}
	}
	if strings.Contains(out, "go_goroutines") {
		t.Error("non-DCGM families from dcgm-exporter must be dropped")
	}
	if strings.Contains(out, "gpuwatch_nccl_busbw_gb_per_second") {
		t.Error("stale textfile must be ignored")
	}
	// Counter reset must not produce a negative rate.
	clock = clock.Add(10 * time.Second)
	write("5", "1")
	if strings.Contains(scrape(t, h), "gpuwatch_net_tx_gbps{") {
		t.Error("counter reset must skip the rate")
	}
}

func TestForceHostnameOverwritesDCGMLabel(t *testing.T) {
	dcgm := serveBytes(t, readFile(t, "testdata/dcgm_exporter_sample.txt"), 200)
	src := &DCGMSource{URL: dcgm.URL, Hostname: "node-x", Force: true, Client: http.DefaultClient}
	fams, err := src.Collect(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var buf strings.Builder
	if err := fams.write(&buf); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(buf.String(), `Hostname="gpu-a"`) || !strings.Contains(buf.String(), `Hostname="node-x"`) {
		t.Errorf("hostname not forced:\n%s", buf.String())
	}
}

func TestTextfileParseErrorMarksSourceDown(t *testing.T) {
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "bad.prom"), []byte("this is { not valid\n"), 0o644)
	h := &Handler{Sources: []Named{{"textfile", &TextfileSource{Dir: dir}}}, Timeout: time.Second}
	if !strings.Contains(scrape(t, h), `gpuwatch_exporter_up{source="textfile"} 0`) {
		t.Error("expected up=0 on parse error")
	}
}

func TestParseFlags(t *testing.T) {
	c, err := parseFlags([]string{"--mode=live", "--listen=:1234"})
	if err != nil || c.Mode != "live" || c.Listen != ":1234" {
		t.Fatalf("got %+v, %v", c, err)
	}
	if _, err := parseFlags([]string{"--mode=bogus"}); err == nil {
		t.Fatal("expected error for bogus mode")
	}
}

func TestRunServesMetricsUntilSignal(t *testing.T) {
	srv := serveBytes(t, readFile(t, "../tests/fixtures/snapshot_healthy.json"), 200)
	errc := make(chan error, 1)
	go func() { errc <- run([]string{"--mode=sim", "--listen=127.0.0.1:19493", "--sim-url=" + srv.URL}) }()
	var body string
	for i := 0; i < 50; i++ {
		resp, err := http.Get("http://127.0.0.1:19493/metrics")
		if err == nil {
			b, _ := io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			body = string(b)
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if !strings.Contains(body, "DCGM_FI_DEV_GPU_TEMP") {
		t.Fatalf("server did not serve metrics: %q", body)
	}
	p, _ := os.FindProcess(os.Getpid())
	_ = p.Signal(os.Interrupt)
	select {
	case err := <-errc:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("server did not shut down")
	}
}
