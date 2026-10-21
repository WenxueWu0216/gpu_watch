import json

import pytest

from bringup.__main__ import main as bringup_main
from bringup.checks import (
    BurnInCheck,
    DriverCheck,
    NcclCheck,
    Status,
    parse_dcgm_diag_json,
    parse_gpu_burn,
    parse_nccl_tests,
    parse_nvidia_smi_csv,
)
from bringup.fakes import FakeKubectl, FakeRunner, FakeSlurm
from bringup.pipeline import HealthGate
from bringup.runner import CommandResult, MutationGuard, SubprocessRunner
from bringup.schedulers import K8sAdapter, NoopAdapter, SlurmAdapter
from bringup.sim_backend import SimNodeRunner
from gpu_sim import Fault

SMI_OK = "\n".join(
    f"{i}, NVIDIA H100 80GB HBM3, GPU-{i}, 550.54.15, 00000000:{0x18 + i:02X}:00.0, Enabled, 0"
    for i in range(8)
)


def const(stdout="", rc=0, stderr=""):
    return lambda argv: CommandResult(tuple(argv), rc, stdout, stderr)


# ------------------------------------------------------------------- parsers
def test_parse_nvidia_smi_csv_handles_na():
    rows = parse_nvidia_smi_csv(
        "0, NVIDIA A10G, GPU-x, 535.104.05, 00000000:00:1E.0, Disabled, [N/A]\n"
    )
    assert rows[0]["ecc_uncorrected"] is None and rows[0]["persistence_mode"] == "Disabled"


def test_parse_gpu_burn():
    assert parse_gpu_burn("blah\n\tGPU 0: OK\n\tGPU 1: FAULTY\n") == {0: True, 1: False}


def test_parse_nccl_tests_rows_and_wrong():
    text = (
        "#       size         count      type   redop    root     time   algbw   busbw #wrong"
        "     time   algbw   busbw #wrong\n"
        "     8388608       2097152     float     sum      -1    100.0   83.89  146.80      0"
        "    101.0   83.06  145.35      2\n"
        "# Avg bus bandwidth    : 146.075 \n"
    )
    p = parse_nccl_tests(text)
    assert p["avg_busbw_gb_s"] == 146.075
    assert p["rows"][0]["busbw_oop"] == 146.80 and p["rows"][0]["busbw_ip"] == 145.35
    assert p["wrong"] == 2


def test_parse_dcgm_diag_nested_and_flat():
    doc = {
        "DCGM GPU Diagnostic": {
            "test_categories": [
                {
                    "category": "Stress",
                    "tests": [
                        {
                            "name": "Targeted Stress",
                            "results": [
                                {"gpu_id": "0", "status": "Pass"},
                                {"gpu_id": "1", "status": "Fail", "warnings": [{"message": "hot"}]},
                            ],
                        }
                    ],
                }
            ]
        }
    }
    rows = parse_dcgm_diag_json(json.dumps(doc))
    assert [(r["test"], r["gpu_id"], r["status"]) for r in rows] == [
        ("Targeted Stress", "0", "Pass"),
        ("Targeted Stress", "1", "Fail"),
    ]
    assert rows[1]["info"] == "hot"
    with pytest.raises(ValueError):
        parse_dcgm_diag_json("not json")


# -------------------------------------------------------------------- checks
def test_driver_check_pass_and_failures():
    assert DriverCheck().run(FakeRunner({"nvidia-smi": const(SMI_OK)})).status is Status.PASS
    r = DriverCheck(expected_gpus=8).run(
        FakeRunner({"nvidia-smi": const(SMI_OK.replace("Enabled, 0\n7", "Enabled, 3\n7"))})
    )
    assert r.status is Status.FAIL and "uncorrectable ECC" in r.summary
    r = DriverCheck(expected_gpus=9).run(FakeRunner({"nvidia-smi": const(SMI_OK)}))
    assert "found 8 GPUs, expected 9" in r.summary
    r = DriverCheck(min_driver="560").run(FakeRunner({"nvidia-smi": const(SMI_OK)}))
    assert r.status is Status.FAIL and "driver 550.54.15 < 560" in r.summary
    r = DriverCheck().run(FakeRunner({"nvidia-smi": const(SMI_OK.replace("Enabled", "Disabled"))}))
    assert r.status is Status.WARN
    r = DriverCheck().run(FakeRunner({"nvidia-smi": const("", 15, "Unable to determine")}))
    assert r.status is Status.FAIL and "rc=15" in r.summary
    r = DriverCheck().run(FakeRunner({}))
    assert r.status is Status.FAIL and "rc=127" in r.summary


def test_nccl_check_threshold_and_errors():
    out = "# Avg bus bandwidth    : 40.0 \n"
    assert NcclCheck(30).run(FakeRunner({"all_reduce_perf": const(out)})).status is Status.PASS
    r = NcclCheck(50).run(FakeRunner({"all_reduce_perf": const(out)}))
    assert r.status is Status.FAIL and "< threshold 50.0" in r.summary
    r = NcclCheck(10).run(FakeRunner({"all_reduce_perf": const("", 1, "NCCL error")}))
    assert r.status is Status.FAIL and "NCCL error" in r.summary
    srun = NcclCheck(10, command_template="srun -N2 all_reduce_perf -g 1")
    assert srun.command()[:2] == ["srun", "-N2"]


def test_burn_in_gpu_burn_and_unknown_tool():
    r = BurnInCheck(tool="gpu_burn").run(FakeRunner({"gpu_burn": const("GPU 0: OK\nGPU 1: OK")}))
    assert r.status is Status.PASS
    r = BurnInCheck(tool="gpu_burn").run(FakeRunner({"gpu_burn": const("", 1)}))
    assert r.status is Status.FAIL
    with pytest.raises(ValueError):
        BurnInCheck(tool="bogus").command()


def test_subprocess_runner_missing_binary_and_timeout():
    r = SubprocessRunner().run(["/nonexistent/definitely-not-here"])
    assert r.returncode == 127
    assert SubprocessRunner().run(["sleep", "5"], timeout_s=0.2).returncode == 124
    assert SubprocessRunner().run(["true"]).ok


# ---------------------------------------------------------------- sim runner
def test_sim_runner_healthy_node_passes_all_checks():
    sim = SimNodeRunner("node-01", seed=1)
    for check in (DriverCheck(), BurnInCheck(), BurnInCheck(tool="gpu_burn"), NcclCheck(30)):
        assert check.run(sim).status is Status.PASS, check.name


@pytest.mark.parametrize(
    ("spec", "failing"),
    [
        ("xid:node-01:4@0", "driver"),
        ("ecc:node-01:2@0,mag=30,variant=dbe", "driver"),
        ("thermal:node-01:3@0,mag=0.3", "burn_in"),
        ("network:node-01@0,mag=0.5", "nccl_allreduce"),
    ],
)
def test_sim_runner_faults_fail_expected_check(spec, failing):
    sim = SimNodeRunner("node-01", faults=[Fault.parse(spec)], seed=1)
    statuses = {c.name: c.run(sim).status for c in (DriverCheck(), BurnInCheck(), NcclCheck(30))}
    assert statuses[failing] is Status.FAIL
    if failing != "driver":
        assert statuses["driver"] is Status.PASS


def test_sim_runner_gpu_burn_misses_thermal_but_dcgm_catches_it():
    # Documented limitation: gpu_burn validates results, not clocks.
    sim = SimNodeRunner("node-01", faults=[Fault.parse("thermal:node-01:3@0,mag=0.3")], seed=1)
    assert BurnInCheck(tool="gpu_burn").run(sim).status is Status.PASS
    r = BurnInCheck(tool="dcgm").run(sim)
    assert r.status is Status.FAIL and "Thermal violations" in r.summary


# ---------------------------------------------------------------- schedulers
def slurm_env(state="IDLE"):
    s = FakeSlurm()
    s.add("gpu-a", state)
    return s, FakeRunner({"scontrol": s})


def test_slurm_drain_dry_run_does_not_mutate():
    s, runner = slurm_env()
    guard = MutationGuard(runner, dry_run=True)
    a = SlurmAdapter(guard)
    assert a.drain("gpu-a", 'driver: 7 GPUs; bad "quote"')
    assert s.nodes["gpu-a"]["State"] == "IDLE"
    assert guard.actions[0].executed is False
    assert guard.actions[0].command.startswith("scontrol update NodeName=gpu-a State=DRAIN")
    assert '"' not in guard.actions[0].command.split("Reason=")[1].strip("'")


def test_slurm_drain_and_resume_execute():
    s, runner = slurm_env()
    a = SlurmAdapter(MutationGuard(runner, dry_run=False))
    assert a.drain("gpu-a", "nccl low")
    assert s.nodes["gpu-a"]["State"] == "IDLE+DRAIN"
    assert a.state("gpu-a") == "IDLE+DRAIN"
    assert a.admit("gpu-a")
    assert s.nodes["gpu-a"]["State"] == "IDLE"


def test_slurm_admit_noop_when_already_idle():
    _, runner = slurm_env()
    guard = MutationGuard(runner, dry_run=False)
    assert SlurmAdapter(guard).admit("gpu-a")
    assert guard.actions == []


def test_slurm_unknown_node():
    _, runner = slurm_env()
    a = SlurmAdapter(MutationGuard(runner, dry_run=False))
    assert a.state("nope").startswith("unknown")
    assert not a.drain("nope", "x")


def test_k8s_drain_cordons_labels_evicts():
    k = FakeKubectl()
    k.add("gpu-a", pods=["trainer-0", "trainer-1"])
    guard = MutationGuard(FakeRunner({"kubectl": k}), dry_run=False)
    a = K8sAdapter(guard)
    assert a.state("gpu-a") == "schedulable,health=none"
    assert a.drain("gpu-a", "burn_in: thermal")
    assert k.nodes["gpu-a"]["spec"]["unschedulable"] is True
    assert k.nodes["gpu-a"]["metadata"]["labels"]["gpuwatch.io/health"] == "failed"
    assert k.pods["gpu-a"] == []
    cmds = [x.command for x in guard.actions]
    assert cmds[0] == "kubectl cordon gpu-a"
    assert cmds[-1].startswith("kubectl drain gpu-a --ignore-daemonsets --delete-emptydir-data")
    assert a.admit("gpu-a")
    assert a.state("gpu-a") == "schedulable,health=passed"


def test_k8s_drain_failure_reported():
    k = FakeKubectl(fail_drain=True)
    k.add("gpu-a", pods=["p"])
    assert not K8sAdapter(MutationGuard(FakeRunner({"kubectl": k}), dry_run=False)).drain(
        "gpu-a", "x"
    )


def test_k8s_dry_run():
    k = FakeKubectl()
    k.add("gpu-a", pods=["p"])
    guard = MutationGuard(FakeRunner({"kubectl": k}), dry_run=True)
    K8sAdapter(guard).drain("gpu-a", "x")
    assert "unschedulable" not in k.nodes["gpu-a"]["spec"] and k.pods["gpu-a"] == ["p"]
    assert len(guard.actions) == 4 and not any(x.executed for x in guard.actions)


# ------------------------------------------------------------------ pipeline
def gate_for(spec=None, scheduler="slurm", dry_run=False, tmp=None):
    sim = SimNodeRunner("node-02", faults=[Fault.parse(spec)] if spec else [], seed=3)
    s, k = FakeSlurm(), FakeKubectl()
    s.add("node-02")
    k.add("node-02", pods=["ddp-0"])
    guard = MutationGuard(FakeRunner({"scontrol": s, "kubectl": k}), dry_run=dry_run)
    adapter = SlurmAdapter(guard) if scheduler == "slurm" else K8sAdapter(guard)
    gate = HealthGate([DriverCheck(), BurnInCheck(), NcclCheck(30)], sim, adapter, tmp)
    return gate, s, k


def test_gate_healthy_node_admitted(tmp_path, otel):
    gate, s, _ = gate_for(tmp=tmp_path)
    r = gate.gate("node-02")
    assert r.passed and r.decision == "admit"
    assert s.nodes["node-02"]["State"] == "IDLE"
    spans = {sp.name for sp in otel[0].get_finished_spans()}
    assert {
        "bringup.gate",
        "bringup.check.driver",
        "bringup.check.burn_in",
        "bringup.check.nccl_allreduce",
    } <= spans


def test_gate_failure_drains_and_writes_report(tmp_path):
    gate, s, _ = gate_for("network:node-02@0,mag=0.5", tmp=tmp_path)
    r = gate.gate("node-02")
    assert not r.passed and r.decision == "drain"
    assert s.nodes["node-02"]["State"] == "IDLE+DRAIN"
    assert "nccl_allreduce" in s.nodes["node-02"]["Reason"]
    report = json.loads((tmp_path / "node-02.json").read_text())
    assert report["diagnosis"][0]["check"] == "nccl_allreduce"
    assert report["diagnosis"][0]["runbook"].startswith("runbook.md#")
    md = (tmp_path / "node-02.md").read_text()
    assert "FAILED" in md and "## Diagnosis" in md and "executed" in md


def test_gate_driver_failure_short_circuits_k8s(tmp_path):
    gate, _, k = gate_for("xid:node-02:1@0", scheduler="k8s", tmp=tmp_path)
    r = gate.gate("node-02")
    assert [c.status for c in r.checks] == [Status.FAIL, Status.SKIPPED, Status.SKIPPED]
    assert k.nodes["node-02"]["spec"]["unschedulable"] is True
    assert r.scheduler_state_after == "cordoned,health=failed"


def test_gate_dry_run_default_reports_without_mutation(tmp_path):
    gate, s, _ = gate_for("ecc:node-02:0@0,variant=dbe", dry_run=True, tmp=tmp_path)
    r = gate.gate("node-02")
    assert r.decision == "drain" and r.dry_run
    assert s.nodes["node-02"]["State"] == "IDLE"
    assert "DRY RUN" in (tmp_path / "node-02.md").read_text()


def test_noop_adapter():
    a = NoopAdapter(MutationGuard(FakeRunner()))
    assert a.drain("x", "y") and a.admit("x") and a.state("x") == "n/a"


# ----------------------------------------------------------------------- CLI
def test_cli_sim_pass_and_fail(tmp_path, capsys):
    rc = bringup_main(["--node", "node-01", "--scheduler", "slurm", "--report-dir", str(tmp_path)])
    assert rc == 0
    rc = bringup_main(
        [
            "--node",
            "node-01",
            "--scheduler",
            "k8s",
            "--report-dir",
            str(tmp_path),
            "--sim-fault",
            "thermal:node-01:5@0,mag=0.3",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 2
    assert "DRY-RUN: kubectl cordon node-01" in out
    assert (tmp_path / "node-01.md").exists()


def test_cli_live_requires_threshold():
    with pytest.raises(SystemExit, match="min-busbw"):
        bringup_main(["--node", "x", "--backend", "live"])
