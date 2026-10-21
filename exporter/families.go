package main

import (
	"io"
	"sort"

	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
	"google.golang.org/protobuf/proto"
)

// Families is a set of metric families keyed by metric name.
type Families map[string]*dto.MetricFamily

// add appends one sample to the family `name`, creating it if needed.
func (f Families) add(name, help string, typ dto.MetricType, value float64, labels map[string]string) {
	mf, ok := f[name]
	if !ok {
		mf = &dto.MetricFamily{Name: proto.String(name), Help: proto.String(help), Type: typ.Enum()}
		f[name] = mf
	}
	m := &dto.Metric{Label: labelPairs(labels)}
	switch typ {
	case dto.MetricType_COUNTER:
		m.Counter = &dto.Counter{Value: proto.Float64(value)}
	default:
		m.Gauge = &dto.Gauge{Value: proto.Float64(value)}
	}
	mf.Metric = append(mf.Metric, m)
}

// merge copies all families of other into f (samples are appended on name collision).
func (f Families) merge(other Families) {
	for name, mf := range other {
		if cur, ok := f[name]; ok {
			cur.Metric = append(cur.Metric, mf.Metric...)
			continue
		}
		f[name] = mf
	}
}

// write renders all families in Prometheus text exposition format, sorted by name.
func (f Families) write(w io.Writer) error {
	names := make([]string, 0, len(f))
	for n := range f {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		if _, err := expfmt.MetricFamilyToText(w, f[n]); err != nil {
			return err
		}
	}
	return nil
}

func labelPairs(labels map[string]string) []*dto.LabelPair {
	keys := make([]string, 0, len(labels))
	for k := range labels {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]*dto.LabelPair, 0, len(keys))
	for _, k := range keys {
		out = append(out, &dto.LabelPair{Name: proto.String(k), Value: proto.String(labels[k])})
	}
	return out
}

// parseText parses Prometheus text exposition format.
func parseText(r io.Reader) (Families, error) {
	var p expfmt.TextParser
	parsed, err := p.TextToMetricFamilies(r)
	if err != nil {
		return nil, err
	}
	return Families(parsed), nil
}

// setLabel sets label name=value on every sample; existing values are
// replaced only when overwrite is true.
func (f Families) setLabel(name, value string, overwrite bool) {
	for _, mf := range f {
		for _, m := range mf.Metric {
			found := false
			for _, lp := range m.Label {
				if lp.GetName() == name {
					found = true
					if overwrite {
						lp.Value = proto.String(value)
					}
					break
				}
			}
			if !found {
				m.Label = append(m.Label, &dto.LabelPair{Name: proto.String(name), Value: proto.String(value)})
			}
		}
	}
}
