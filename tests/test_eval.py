from eval.run import (
    README_BEGIN,
    README_END,
    RunResult,
    _run_one,
    check_gate,
    load_config,
    render_markdown,
    run_eval,
    summarize,
    update_readme,
)

THRESH = {
    "min_detection_rate": {
        "thermal": 0.9,
        "xid": 0.9,
        "ecc": 0.9,
        "straggler": 0.5,
        "network": 0.9,
    },
    "max_healthy_fp_rate": 0.2,
}


def small_cfg():
    cfg = load_config()
    cfg.update(runs_per_fault=2, healthy_runs=2, duration_s=900)
    return cfg


def test_run_one_is_deterministic():
    cfg = small_cfg()
    assert _run_one(("straggler", 5, cfg)) == _run_one(("straggler", 5, cfg))


def test_run_one_xid_detected_quickly():
    r = _run_one(("xid", 1, small_cfg()))
    assert r.detected and r.ttd_s is not None and 0 <= r.ttd_s <= 10


def test_small_eval_end_to_end(tmp_path):
    cfg = small_cfg()
    results = run_eval(cfg, workers=2)
    assert len(results) == 5 * 2 + 2
    summary = summarize(results, cfg["duration_s"])
    assert set(summary["per_fault"]) == {"thermal", "xid", "ecc", "straggler", "network"}
    md = render_markdown(summary, cfg, THRESH, check_gate(summary, THRESH))
    assert "SIMULATED" in md and "| thermal |" in md


def _fake(kind, detected, kinds=()):
    return RunResult(
        kind, 0, "", 1.0, "", detected, 10.0 if detected else None, len(kinds), tuple(kinds)
    )


def test_summary_rates_and_gate():
    rs = [_fake("thermal", True), _fake("thermal", False)]
    rs += [_fake(k, True) for k in ("xid", "ecc", "straggler", "network")]
    rs += [_fake("healthy", False), _fake("healthy", False, ["straggler"])]
    s = summarize(rs, 1800)
    assert s["per_fault"]["thermal"]["detection_rate"] == 0.5
    assert s["per_fault"]["straggler"]["healthy_fp_rate"] == 0.5
    assert s["per_fault"]["xid"]["healthy_fp_rate"] == 0.0
    assert s["healthy"]["any_event_rate"] == 0.5
    assert s["healthy"]["false_events_per_sim_hour"] == 1.0
    fails = check_gate(s, THRESH)
    assert any("thermal" in f for f in fails)
    assert any("healthy FP" in f for f in fails)


def test_update_readme_replaces_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{README_BEGIN}\nOLD\n{README_END}\noutro\n")
    rs = [_fake(k, True) for k in ("thermal", "xid", "ecc", "straggler", "network")]
    rs.append(_fake("healthy", False))
    assert update_readme(readme, summarize(rs, 1800))
    text = readme.read_text()
    assert "OLD" not in text and "| thermal | 1 | 100.0% (1/1)" in text
    assert text.startswith("intro\n") and text.endswith("outro\n")
    assert not update_readme(tmp_path / "missing.md", summarize(rs, 1800))
