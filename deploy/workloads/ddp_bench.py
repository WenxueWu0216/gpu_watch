"""Synthetic PyTorch data-parallel benchmark that exports GPUWatch job metrics.

Launch with torchrun (one process per GPU)::

    torchrun --nnodes 2 --nproc-per-node 8 --rdzv-backend c10d \
        --rdzv-endpoint $MASTER:29500 deploy/workloads/ddp_bench.py --steps 2000

Each step runs forward+backward with gradient sync disabled (``no_sync``) and
then all-reduces one flat gradient bucket explicitly. That sacrifices
compute/communication overlap, but yields the two per-rank quantities the
detector needs:

* time-to-barrier (forward + backward + input pipeline) ->
  ``gpuwatch_train_step_time_seconds{rank=...}``
* all-reduce bus bandwidth (nccl-tests definition, 2(n-1)/n * bytes / time) ->
  ``gpuwatch_nccl_busbw_gb_per_second{rank=...}``

Both are written atomically as a Prometheus textfile
(``<textfile-dir>/gpuwatch_ddp_rank<R>.prom``) which the Go exporter merges in
live mode. ``--slow-rank``/``--slow-factor`` and ``--host-delay-ms`` inject a
GPU-bound or host-bound straggler for end-to-end testing on real hardware.

``--device cpu`` uses the gloo backend so the script can be smoke-tested
without GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn


def build_model(hidden: int, layers: int) -> nn.Module:
    """A stack of wide MLP blocks: compute heavy with a large gradient payload."""
    blocks: list[nn.Module] = []
    for _ in range(layers):
        blocks += [nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden)]
    return nn.Sequential(*blocks)


def write_textfile(
    path: Path, host: str, rank: int, local_rank: int, step_s: float, busbw: float, steps: int
) -> None:
    """Atomically write the rank's metrics in Prometheus text format."""
    lbl = f'rank="{rank}",local_rank="{local_rank}",Hostname="{host}"'
    body = (
        "# HELP gpuwatch_train_step_time_seconds Rank time-to-barrier (fwd+bwd+input), "
        "median of recent steps.\n"
        "# TYPE gpuwatch_train_step_time_seconds gauge\n"
        f"gpuwatch_train_step_time_seconds{{{lbl}}} {step_s:.6f}\n"
        "# HELP gpuwatch_nccl_busbw_gb_per_second All-reduce bus bandwidth (GB/s).\n"
        "# TYPE gpuwatch_nccl_busbw_gb_per_second gauge\n"
        f"gpuwatch_nccl_busbw_gb_per_second{{{lbl}}} {busbw:.3f}\n"
        "# HELP gpuwatch_train_steps_total Steps completed by this rank.\n"
        "# TYPE gpuwatch_train_steps_total counter\n"
        f"gpuwatch_train_steps_total{{{lbl}}} {steps}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(body)
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--textfile-dir", default="/var/lib/gpuwatch/textfile")
    p.add_argument(
        "--hostname",
        default=os.environ.get("GPUWATCH_HOSTNAME", socket.gethostname()),
        help="Hostname label; must match the node name used by the exporter",
    )
    p.add_argument("--report-every", type=int, default=10)
    p.add_argument("--window", type=int, default=20, help="steps in the reported median")
    p.add_argument("--slow-rank", type=int, default=-1)
    p.add_argument(
        "--slow-factor",
        type=float,
        default=1.0,
        help="extra GPU work on --slow-rank (1.3 = 30%% more compute)",
    )
    p.add_argument(
        "--host-delay-ms",
        type=float,
        default=0.0,
        help="sleep per step on --slow-rank (host-bound straggler)",
    )
    args = p.parse_args()

    cuda = args.device == "cuda"
    dist.init_process_group("nccl" if cuda else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
    torch.manual_seed(rank)
    model = build_model(args.hidden, args.layers).to(device)
    ddp = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if cuda else None)
    opt = torch.optim.SGD(ddp.parameters(), lr=1e-4)
    x = torch.randn(args.batch, args.hidden, device=device)
    params = [q for q in model.parameters() if q.requires_grad]
    nbytes = sum(q.numel() * q.element_size() for q in params)
    textfile = Path(args.textfile_dir) / f"gpuwatch_ddp_rank{rank}.prom"
    slow = rank == args.slow_rank
    compute_hist: list[float] = []
    bw_hist: list[float] = []

    def sync() -> None:
        if cuda:
            torch.cuda.synchronize(device)

    for step in range(1, args.steps + 1):
        sync()
        t0 = time.perf_counter()
        if slow and args.host_delay_ms > 0:
            time.sleep(args.host_delay_ms / 1000.0)
        with ddp.no_sync():
            loss = ddp(x).float().pow(2).mean()
            loss.backward()
            if slow and args.slow_factor > 1.0:
                extra = int((args.slow_factor - 1.0) * 10)
                for _ in range(max(1, extra)):
                    model(x).sum().backward()
        sync()
        t1 = time.perf_counter()
        flat = torch.cat([q.grad.reshape(-1) for q in params if q.grad is not None])
        # Barrier first so the timed all-reduce excludes waiting for slower
        # ranks (that skew is what step_time captures).
        dist.barrier()
        sync()
        t2 = time.perf_counter()
        dist.all_reduce(flat)
        sync()
        t3 = time.perf_counter()
        flat /= world
        offset = 0
        for q in params:
            if q.grad is not None:
                n = q.numel()
                q.grad.copy_(flat[offset : offset + n].view_as(q.grad))
                offset += n
        opt.step()
        opt.zero_grad(set_to_none=True)
        compute_hist = [*compute_hist, t1 - t0][-args.window :]
        comm = max(t3 - t2, 1e-9)
        busbw = 2 * (world - 1) / world * nbytes / comm / 1e9 if world > 1 else 0.0
        bw_hist = [*bw_hist, busbw][-args.window :]
        if step % args.report_every == 0:
            med_step = statistics.median(compute_hist)
            med_bw = statistics.median(bw_hist)
            write_textfile(textfile, args.hostname, rank, local_rank, med_step, med_bw, step)
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "rank": rank,
                            "step_time_s": round(med_step, 5),
                            "busbw_gb_s": round(med_bw, 3),
                            "grad_mb": nbytes / 2**20,
                        }
                    ),
                    flush=True,
                )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
