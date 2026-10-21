# GPUWatch evaluation results (SIMULATED)

> **All numbers on this page come from the deterministic simulator in `gpu_sim/`,
> not from real GPUs.** They measure the detector against the simulator's fault
> model; they are not a claim about production accuracy. Regenerate with `make eval`.

## Protocol

- Cluster: 8 nodes x 8 GPUs, 10 s scrape interval, 1800 simulated seconds per run.
- 100 runs per fault type, one fault per run, seeds 0..99; fault target, onset (uniform in [300, 900] s) and severity drawn per seed.
- 200 healthy runs (100 simulated hours) with benign noise (jitter, GC spikes, multi-scrape slowdowns, checkpoint pauses, power capping, sporadic SBEs).
- Detector thresholds (`detector/config.yaml`) were tuned on seeds >= 1000, disjoint from these.
- Matching rules are documented in `eval/run.py`.

## Results per fault type

| Fault type (simulated) | Runs | Detection rate | Healthy-run FP rate | TTD median (sim s) | TTD p90 (sim s) | TTD max (sim s) |
|---|---:|---:|---:|---:|---:|---:|
| thermal | 100 | 100.0% (100/100) | 0.0% (0/200) | 103.9 | 136.6 | 172.2 |
| xid | 100 | 100.0% (100/100) | 0.0% (0/200) | 5.3 | 8.8 | 9.8 |
| ecc | 100 | 99.0% (99/100) | 0.0% (0/200) | 5.7 | 16.2 | 70.6 |
| straggler | 100 | 92.0% (92/100) | 2.5% (5/200) | 75.2 | 112.5 | 1152.5 |
| network | 100 | 99.0% (99/100) | 0.0% (0/200) | 45.1 | 49.7 | 377.4 |

Healthy runs with >= 1 event of any kind (simulated): 5/200 (2.5%); 6 false events total = 0.06 per simulated hour.

## Detection by fault severity (simulated)

Severity ranges intentionally extend below the detector's sensitivity floor (e.g. 5% step-time slowdown) so the misses are visible rather than hidden.

**thermal**

| Bucket | Runs | Detected | Rate | TTD median (sim s) |
|---|---:|---:|---:|---:|
| clock -2..8% | 16 | 16 | 100.0% | 141.1 |
| clock -8..15% | 17 | 17 | 100.0% | 123 |
| clock -15..35% | 67 | 67 | 100.0% | 93.9 |

**ecc**

| Bucket | Runs | Detected | Rate | TTD median (sim s) |
|---|---:|---:|---:|---:|
| SBE <2/scrape | 4 | 3 | 75.0% | 28.7 |
| SBE 2..10/scrape | 14 | 14 | 100.0% | 14.8 |
| SBE >=10/scrape | 82 | 82 | 100.0% | 5.2 |
| variant=dbe | 47 | 47 | 100.0% | 5.7 |
| variant=sbe | 53 | 52 | 98.1% | 5.6 |

**straggler**

| Bucket | Runs | Detected | Rate | TTD median (sim s) |
|---|---:|---:|---:|---:|
| slowdown 1-5% | 12 | 4 | 33.3% | 106.2 |
| slowdown 5-10% | 16 | 16 | 100.0% | 113.8 |
| slowdown 10-20% | 20 | 20 | 100.0% | 73.9 |
| slowdown 20-40% | 52 | 52 | 100.0% | 73.1 |
| variant=gpu | 55 | 52 | 94.5% | 75.5 |
| variant=host | 45 | 40 | 88.9% | 73.7 |

**network**

| Bucket | Runs | Detected | Rate | TTD median (sim s) |
|---|---:|---:|---:|---:|
| bw x0.90-0.98 | 11 | 10 | 90.9% | 84 |
| bw x0.70-0.90 | 23 | 23 | 100.0% | 44.7 |
| bw x0.30-0.70 | 66 | 66 | 100.0% | 45 |

## Off-target events in faulted runs (simulated)

| Fault type | Off-target firing events |
|---|---:|
| thermal | 4 |
| xid | 1 |
| ecc | 3 |
| straggler | 3 |
| network | 3 |

## Healthy-run false positives (simulated)

| Seed | Event kinds |
|---:|---|
| 10007 | straggler |
| 10039 | straggler |
| 10154 | straggler |
| 10173 | straggler |
| 10190 | straggler |

## Missed faults (simulated)

| Type | Seed | Fault |
|---|---:|---|
| ecc | 10 | `ecc@node-01/gpu4 t=554s mag=0.314 variant=sbe` |
| straggler | 16 | `straggler@node-03 t=617s mag=1.02 variant=gpu` |
| straggler | 45 | `straggler@node-04 t=672s mag=1.02 variant=host` |
| straggler | 53 | `straggler@node-05 t=683s mag=1.01 variant=host` |
| straggler | 64 | `straggler@node-01 t=716s mag=1.03 variant=host` |
| straggler | 67 | `straggler@node-05 t=720s mag=1.02 variant=gpu` |
| straggler | 74 | `straggler@node-06 t=516s mag=1.02 variant=gpu` |
| straggler | 82 | `straggler@node-01 t=475s mag=1.02 variant=host` |
| straggler | 83 | `straggler@node-04 t=686s mag=1.03 variant=host` |
| network | 81 | `network@node-07 t=369s mag=0.978` |

## CI gate

Thresholds from `eval/thresholds.yaml`:

| Fault type | Min detection rate | Measured |
|---|---:|---:|
| thermal | 95% | 100.0% |
| xid | 99% | 100.0% |
| ecc | 95% | 99.0% |
| straggler | 85% | 92.0% |
| network | 95% | 99.0% |

Max healthy-run FP rate (any event): 5%; measured 2.5%.

Gate status: **PASS**
