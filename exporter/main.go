// Command gpuwatch-exporter publishes GPU and job telemetry for Prometheus.
//
// Modes:
//
//	--mode=sim   poll the Python simulator (gpu_sim serve) and expose its
//	             snapshot under DCGM metric names.
//	--mode=live  proxy one dcgm-exporter endpoint, merge workload textfiles
//	             (*.prom) and InfiniBand sysfs counters.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// Config holds parsed command-line options.
type Config struct {
	Mode        string
	Listen      string
	SimURL      string
	DCGMURL     string
	TextfileDir string
	IBRoot      string
	Hostname    string
	ForceHost   bool
	Timeout     time.Duration
}

func parseFlags(args []string) (Config, error) {
	host, _ := os.Hostname()
	var c Config
	fs := flag.NewFlagSet("gpuwatch-exporter", flag.ContinueOnError)
	fs.StringVar(&c.Mode, "mode", "sim", "sim | live")
	fs.StringVar(&c.Listen, "listen", ":9400", "listen address")
	fs.StringVar(&c.SimURL, "sim-url", "http://localhost:9500/snapshot", "simulator snapshot URL (sim mode)")
	fs.StringVar(&c.DCGMURL, "dcgm-url", "http://localhost:9401/metrics", "dcgm-exporter URL (live mode)")
	fs.StringVar(&c.TextfileDir, "textfile-dir", "/var/lib/gpuwatch/textfile", "directory of *.prom files (live mode; empty disables)")
	fs.StringVar(&c.IBRoot, "ib-root", "/sys/class/infiniband", "InfiniBand sysfs root (live mode; empty disables)")
	fs.StringVar(&c.Hostname, "hostname", host, "Hostname label for series that lack one")
	fs.BoolVar(&c.ForceHost, "force-hostname", false, "overwrite Hostname on all live series with --hostname")
	fs.DurationVar(&c.Timeout, "timeout", 5*time.Second, "per-source collection timeout")
	if err := fs.Parse(args); err != nil {
		return c, err
	}
	if c.Mode != "sim" && c.Mode != "live" {
		return c, fmt.Errorf("unknown --mode %q", c.Mode)
	}
	return c, nil
}

// BuildCollectors returns the named sources for the configured mode.
func BuildCollectors(c Config) []Named {
	client := &http.Client{Timeout: c.Timeout}
	if c.Mode == "sim" {
		return []Named{{"sim", &SimSource{URL: c.SimURL, Client: client}}}
	}
	out := []Named{{"dcgm", &DCGMSource{URL: c.DCGMURL, Hostname: c.Hostname, Force: c.ForceHost, Client: client}}}
	if c.TextfileDir != "" {
		out = append(out, Named{"textfile", &TextfileSource{Dir: c.TextfileDir, Hostname: c.Hostname, Force: c.ForceHost, MaxAge: 2 * time.Minute}})
	}
	if c.IBRoot != "" {
		if _, err := os.Stat(c.IBRoot); err == nil {
			out = append(out, Named{"infiniband", &InfinibandSource{Root: c.IBRoot, Hostname: c.Hostname}})
		}
	}
	return out
}

func run(args []string) error {
	cfg, err := parseFlags(args)
	if err != nil {
		return err
	}
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	h := &Handler{Sources: BuildCollectors(cfg), Timeout: cfg.Timeout, Log: logger}
	mux := http.NewServeMux()
	mux.Handle("/metrics", h)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprintln(w, "ok")
	})
	srv := &http.Server{Addr: cfg.Listen, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdown)
	}()
	logger.Info("gpuwatch-exporter listening", "addr", cfg.Listen, "mode", cfg.Mode)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		return err
	}
	return nil
}
