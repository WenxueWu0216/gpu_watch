import json
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from ai_diagnosis.__main__ import FIXTURES, incident_from_sim
from ai_diagnosis.__main__ import main as diag_main
from ai_diagnosis.diagnose import (
    OUTPUT_SCHEMA,
    SYSTEM_PROMPT,
    AnthropicBackend,
    BackendError,
    Diagnoser,
    RecordingBackend,
    ReplayBackend,
    _numbers,
    build_user_prompt,
    prompt_sha,
    validate,
)
from ai_diagnosis.incident import EvidenceItem, Incident
from detector.events import Event, EventKind, Severity

ALL_FIXTURES = sorted(FIXTURES.glob("*.json"))


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def fixture_incident(name: str) -> Incident:
    fx = load(FIXTURES / name)
    sc = fx["scenario"]
    inc = incident_from_sim(sc["faults"], sc["seed"], sc["kind"])
    assert inc is not None
    return inc


@pytest.mark.parametrize("path", ALL_FIXTURES, ids=lambda p: p.stem)
def test_fixtures_match_current_incidents_and_prompt(path):
    """Fixtures must be regenerable: same incident ID, evidence and prompt."""
    fx = load(path)
    assert "NOT recorded from the Anthropic API" in fx["provenance"]
    inc = incident_from_sim(
        fx["scenario"]["faults"], fx["scenario"]["seed"], fx["scenario"]["kind"]
    )
    assert inc is not None and inc.id == fx["incident_id"]
    assert inc.evidence_sha() == fx["evidence_sha"]
    assert prompt_sha(SYSTEM_PROMPT, build_user_prompt(inc)) == fx["prompt_sha"]


def diagnose(name):
    inc = fixture_incident(name)
    backend = ReplayBackend(FIXTURES)
    d = Diagnoser(backend).diagnose(inc)
    assert backend.stale == []
    return inc, d


def test_grounded_thermal_diagnosis():
    inc, d = diagnose("thermal_throttle_node-03_2_400.json")
    assert d.status == "diagnosed" and d.grounded
    assert "thermal" in d.hypothesis.lower()
    ids = set(inc.by_id())
    assert all(c.citations and set(c.citations) <= ids for c in d.claims)
    assert d.rejected == []
    md = d.render_markdown(inc)
    assert "`M2`: DCGM_FI_DEV_SM_CLOCK" in md


def test_xid_diagnosis_cites_kernel_log():
    inc, d = diagnose("xid_error_node-05_6_300.json")
    assert d.status == "diagnosed"
    assert "L1" in d.hypothesis_citations
    assert "fallen off the bus" in inc.by_id()["L1"].text


def test_model_declines_when_evidence_is_insufficient():
    _, d = diagnose("straggler_node-01_None_370.json")
    assert d.status == "insufficient_evidence"
    assert d.hypothesis == "insufficient evidence"
    assert any("data-loader" in x for x in d.missing_evidence)


def test_invented_number_claim_is_rejected_but_rest_kept():
    _, d = diagnose("ecc_error_node-02_4_300.json")
    assert d.status == "diagnosed"
    assert len(d.claims) == 3
    assert len(d.rejected) == 1 and "240" in d.rejected[0]["reason"]


def test_fully_ungrounded_output_becomes_insufficient():
    _, d = diagnose("nccl_slow_link_node-06_None_340.json")
    assert d.status == "insufficient_evidence"
    reasons = " ".join(r["reason"] for r in d.rejected)
    assert "unknown evidence ['L7']" in reasons and "no citation" in reasons
    assert any("grounding check failed" in n for n in d.notes)


def tiny_incident(n_items=4):
    ev = Event(10, EventKind.THERMAL, Severity.WARNING, "n1", 0, "hot")
    items = [
        EvidenceItem("M1", "metric", "DCGM_FI_DEV_GPU_TEMP{node=n1,gpu=0} last=91.5"),
        EvidenceItem("M2", "metric", "DCGM_FI_DEV_SM_CLOCK{node=n1,gpu=0} last=1500"),
        EvidenceItem("M3", "metric", "peer median clock=1980"),
        EvidenceItem("L1", "log", "nothing"),
    ][:n_items]
    return Incident("t", ev, items)


class Never:
    name = "never"

    def complete(self, *a):
        raise AssertionError("backend must not be called")


def test_precheck_skips_model_when_too_little_evidence():
    d = Diagnoser(Never()).diagnose(tiny_incident(2))
    assert d.status == "insufficient_evidence" and d.backend == "precheck"


def good_response(**over):
    r = {
        "status": "diagnosed",
        "hypothesis": "GPU 0 is thermally throttling",
        "hypothesis_citations": ["M1", "M2"],
        "confidence": "high",
        "claims": [
            {"text": "Temperature reached 91.5 C", "citations": ["M1"]},
            {"text": "Clock is 1500 MHz vs peers at 1980", "citations": ["M2", "M3"]},
        ],
        "next_action": "check cooling",
        "next_action_citations": ["M1"],
        "missing_evidence": [],
    }
    r.update(over)
    return json.dumps(r)


def test_validate_numbers_must_come_from_cited_items():
    inc = tiny_incident()
    assert validate(inc, good_response()).status == "diagnosed"
    bad = good_response(
        claims=[{"text": "Clock is 1500 MHz vs peers at 1980", "citations": ["M2"]}]
    )  # 1980 only in M3
    d = validate(inc, bad)
    assert d.status == "insufficient_evidence" and "1980" in d.rejected[0]["reason"]
    d = validate(inc, good_response(hypothesis="GPU hit 99.9 C", hypothesis_citations=["M1"]))
    assert d.status == "insufficient_evidence"
    assert validate(inc, "not json").status == "insufficient_evidence"
    d = validate(inc, good_response(next_action_citations=["X9"]))
    assert d.status == "insufficient_evidence"


def test_number_normalization():
    assert _numbers("at 88.650 C on gpu 2 node-03 0x20") == {"88.65", "03"}


class FakeMessages:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.kwargs = resp, exc, None

    def create(self, **kw):
        self.kwargs = kw
        if self.exc:
            raise self.exc
        return self.resp


def fake_client(text=None, stop="end_turn", exc=None):
    content = (
        [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)]
        if text
        else []
    )
    msgs = FakeMessages(SimpleNamespace(stop_reason=stop, content=content), exc)
    return SimpleNamespace(beta=SimpleNamespace(messages=msgs)), msgs


def test_anthropic_backend_request_shape():
    client, msgs = fake_client(good_response())
    d = Diagnoser(AnthropicBackend(client=client)).diagnose(tiny_incident())
    assert d.status == "diagnosed" and d.backend == "anthropic"
    kw = msgs.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["output_config"]["format"] == {"type": "json_schema", "schema": OUTPUT_SCHEMA}
    assert kw["system"] == SYSTEM_PROMPT
    assert "[M1] (metric)" in kw["messages"][0]["content"]
    assert "thinking" not in kw  # thinking cannot be disabled on this model; left adaptive


@pytest.mark.parametrize(("stop", "msg"), [("refusal", "declined"), ("max_tokens", "truncated")])
def test_anthropic_backend_bad_stop_reason(stop, msg):
    client, _ = fake_client(good_response(), stop=stop)
    d = Diagnoser(AnthropicBackend(client=client)).diagnose(tiny_incident())
    assert d.status == "insufficient_evidence" and msg in d.notes[0]


def test_anthropic_backend_connection_error():
    exc = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x"))
    client, _ = fake_client(exc=exc)
    d = Diagnoser(AnthropicBackend(client=client)).diagnose(tiny_incident())
    assert d.status == "insufficient_evidence" and "connection" in d.notes[0]


def test_recording_then_replay_roundtrip(tmp_path):
    client, _ = fake_client(good_response())
    inc = tiny_incident()
    rec = RecordingBackend(AnthropicBackend(client=client), tmp_path)
    assert Diagnoser(rec).diagnose(inc).status == "diagnosed"
    replay = ReplayBackend(tmp_path)
    assert Diagnoser(replay).diagnose(inc).status == "diagnosed"
    assert replay.stale == []
    assert "recorded from anthropic" in load(next(tmp_path.glob("*.json")))["provenance"]


def test_replay_missing_fixture(tmp_path):
    with pytest.raises(BackendError):
        ReplayBackend(tmp_path).complete(tiny_incident(), "s", "u")
    d = Diagnoser(ReplayBackend(tmp_path)).diagnose(tiny_incident())
    assert d.status == "insufficient_evidence" and "no replay fixture" in d.notes[0]


def test_cli_replay(capsys):
    assert diag_main(["--fault", "thermal:node-03:2@300,mag=0.3", "--seed", "7", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "diagnosed"
    assert diag_main(["--seed", "1"]) in (0, 1)  # healthy run: usually no alert
