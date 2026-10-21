"""LLM incident summarizer with a hard grounding check.

The model must return JSON (enforced with structured outputs on the Anthropic
backend) in which the hypothesis, every claim and the next action cite
evidence IDs from the incident. :func:`validate` then enforces grounding
independently of the model:

* every cited ID must exist in the incident;
* every claim needs >= 1 valid citation;
* every multi-digit number in a claim must appear in the text of one of the
  evidence items it cites (no invented measurements);
* the hypothesis and next action must cite valid evidence.

If the model reports ``insufficient_evidence``, if too little evidence was
collected to begin with, or if the output fails the grounding check, the
result is ``insufficient_evidence`` - the system never falls back to an
uncited guess.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ai_diagnosis.incident import Incident

PROMPT_VERSION = "v1"
DEFAULT_MODEL = "claude-opus-5-5"
MIN_EVIDENCE_ITEMS = 3

SYSTEM_PROMPT = """You are an SRE assistant diagnosing GPU cluster incidents.

You receive one alert and a numbered list of evidence items (alert evidence A*, metrics M*, \
log lines L*). Produce a root-cause hypothesis, the supporting claims, and one concrete next \
action.

Grounding rules (a validator enforces them; violating output is discarded):
- Use only the evidence provided. Do not rely on outside assumptions about this cluster.
- Every claim must cite the IDs of the evidence items that support it, e.g. ["M2", "L1"].
- Only quote numbers that appear verbatim in the evidence items you cite.
- The hypothesis and the next action must also cite evidence.
- If the evidence does not clearly support a single root cause, set status to \
"insufficient_evidence", leave the hypothesis as "insufficient evidence", and list what \
additional evidence would be needed in missing_evidence. Never guess."""

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "status",
        "hypothesis",
        "hypothesis_citations",
        "confidence",
        "claims",
        "next_action",
        "next_action_citations",
        "missing_evidence",
    ],
    "properties": {
        "status": {"type": "string", "enum": ["diagnosed", "insufficient_evidence"]},
        "hypothesis": {"type": "string"},
        "hypothesis_citations": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "citations"],
                "properties": {
                    "text": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "next_action": {"type": "string"},
        "next_action_citations": {"type": "array", "items": {"type": "string"}},
        "missing_evidence": {"type": "array", "items": {"type": "string"}},
    },
}


class BackendError(RuntimeError):
    """The backend could not produce a response (refusal, missing fixture, API error)."""


class Backend(Protocol):
    """Produces raw JSON text for a prompt."""

    name: str

    def complete(self, incident: Incident, system: str, user: str) -> str:
        """Return the model's JSON output as text."""
        ...


def build_user_prompt(incident: Incident) -> str:
    """The user turn: the rendered incident."""
    return (
        f"{incident.render()}\n\nReturn the diagnosis JSON for this incident. "
        "Cite evidence IDs exactly as written in brackets."
    )


def prompt_sha(system: str, user: str) -> str:
    """Hash identifying an exact prompt (for fixture staleness checks)."""
    return hashlib.sha256(f"{PROMPT_VERSION}\n{system}\n{user}".encode()).hexdigest()[:16]


@dataclass
class Claim:
    """A grounded statement."""

    text: str
    citations: list[str]


@dataclass
class Diagnosis:
    """Validated diagnosis output."""

    incident_id: str
    status: str
    hypothesis: str
    confidence: str
    claims: list[Claim] = field(default_factory=list)
    hypothesis_citations: list[str] = field(default_factory=list)
    next_action: str = ""
    next_action_citations: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    backend: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        """True if diagnosed and every kept claim passed validation."""
        return self.status == "diagnosed"

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict."""
        return asdict(self)

    def render_markdown(self, incident: Incident) -> str:
        """Human-readable summary with cited evidence inline."""
        ev = incident.by_id()
        out = [
            f"## Incident {self.incident_id}",
            "",
            f"- Status: **{self.status}** (backend: {self.backend})",
        ]
        if self.status == "diagnosed":
            out += [
                f"- Hypothesis ({self.confidence} confidence): {self.hypothesis} "
                f"[{', '.join(self.hypothesis_citations)}]",
                "",
                "### Evidence",
                "",
            ]
            for c in self.claims:
                out.append(f"- {c.text} [{', '.join(c.citations)}]")
                for cid in c.citations:
                    out.append(f"  - `{cid}`: {ev[cid].text}")
            out += [
                "",
                "### Suggested next action",
                "",
                f"{self.next_action} [{', '.join(self.next_action_citations)}]",
            ]
        else:
            out.append("- Hypothesis: insufficient evidence")
            if self.missing_evidence:
                out += ["", "### Evidence needed", ""] + [f"- {x}" for x in self.missing_evidence]
        if self.rejected:
            out += ["", "### Rejected (ungrounded) model output", ""]
            out += [f"- {r['text']!r}: {r['reason']}" for r in self.rejected]
        for n in self.notes:
            out.append(f"\n> {n}")
        return "\n".join(out) + "\n"


_NUM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])")


def _numbers(text: str) -> set[str]:
    """Multi-digit numbers in ``text`` (single digits are too ambiguous to check)."""
    out = set()
    for n in _NUM_RE.findall(text):
        if len(n.replace(".", "")) >= 2:
            out.add(n.rstrip("0").rstrip(".") if "." in n else n)
    return out


def _norm_numbers(text: str) -> set[str]:
    return {n.rstrip("0").rstrip(".") if "." in n else n for n in _NUM_RE.findall(text)}


def validate(incident: Incident, raw: str, backend: str = "") -> Diagnosis:
    """Parse model output and enforce the grounding rules (see module docstring)."""
    ev = incident.by_id()
    insufficient = Diagnosis(
        incident.id, "insufficient_evidence", "insufficient evidence", "low", backend=backend
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        insufficient.notes.append(f"model output was not valid JSON: {exc}")
        return insufficient
    insufficient.missing_evidence = [str(x) for x in data.get("missing_evidence", [])]
    if data.get("status") != "diagnosed":
        insufficient.notes.append("model reported insufficient evidence")
        return insufficient

    def valid_ids(ids: Any) -> list[str]:
        return [str(i) for i in (ids or []) if str(i) in ev]

    kept: list[Claim] = []
    rejected: list[dict[str, Any]] = []
    for c in data.get("claims", []):
        text, cites = str(c.get("text", "")), [str(x) for x in c.get("citations", [])]
        unknown = [x for x in cites if x not in ev]
        good = valid_ids(cites)
        cited_numbers: set[str] = set()
        for cid in good:
            cited_numbers |= _norm_numbers(ev[cid].text)
        invented = sorted(_numbers(text) - cited_numbers)
        if unknown:
            rejected.append({"text": text, "reason": f"cites unknown evidence {unknown}"})
        elif not good:
            rejected.append({"text": text, "reason": "no citation"})
        elif invented:
            rejected.append(
                {"text": text, "reason": f"numbers {invented} not in cited evidence {good}"}
            )
        else:
            kept.append(Claim(text, good))
    hyp_cites = valid_ids(data.get("hypothesis_citations"))
    act_cites = valid_ids(data.get("next_action_citations"))
    problems = []
    if not kept:
        problems.append("no claim passed the grounding check")
    if not hyp_cites:
        problems.append("hypothesis has no valid citation")
    if not act_cites:
        problems.append("next action has no valid citation")
    hyp_invented = _numbers(str(data.get("hypothesis", ""))) - {
        n for cid in hyp_cites for n in _norm_numbers(ev[cid].text)
    }
    if hyp_invented:
        problems.append(f"hypothesis quotes numbers {sorted(hyp_invented)} not in its citations")
    if problems:
        insufficient.rejected = rejected
        insufficient.notes.append("grounding check failed: " + "; ".join(problems))
        return insufficient
    return Diagnosis(
        incident_id=incident.id,
        status="diagnosed",
        hypothesis=str(data["hypothesis"]),
        confidence=str(data.get("confidence", "low")),
        claims=kept,
        hypothesis_citations=hyp_cites,
        next_action=str(data.get("next_action", "")),
        next_action_citations=act_cites,
        missing_evidence=[str(x) for x in data.get("missing_evidence", [])],
        rejected=rejected,
        backend=backend,
    )


@dataclass
class Diagnoser:
    """Runs a backend on an incident and validates the result."""

    backend: Backend
    min_evidence: int = MIN_EVIDENCE_ITEMS

    def diagnose(self, incident: Incident) -> Diagnosis:
        """Diagnose ``incident``; never returns an ungrounded hypothesis."""
        from tracing.otel import get_tracer

        with get_tracer("gpuwatch.ai_diagnosis").start_as_current_span("ai_diagnosis") as span:
            span.set_attribute("gpuwatch.incident", incident.id)
            span.set_attribute("gpuwatch.backend", self.backend.name)
            if len(incident.items) < self.min_evidence:
                d = Diagnosis(
                    incident.id,
                    "insufficient_evidence",
                    "insufficient evidence",
                    "low",
                    backend="precheck",
                    missing_evidence=["more correlated metrics or logs"],
                    notes=[
                        f"only {len(incident.items)} evidence items collected "
                        f"(< {self.min_evidence}); model not called"
                    ],
                )
            else:
                system, user = SYSTEM_PROMPT, build_user_prompt(incident)
                try:
                    raw = self.backend.complete(incident, system, user)
                except BackendError as exc:
                    d = Diagnosis(
                        incident.id,
                        "insufficient_evidence",
                        "insufficient evidence",
                        "low",
                        backend=self.backend.name,
                        notes=[str(exc)],
                    )
                else:
                    d = validate(incident, raw, self.backend.name)
            span.set_attribute("gpuwatch.diagnosis.status", d.status)
            span.set_attribute("gpuwatch.diagnosis.rejected_claims", len(d.rejected))
            return d


class AnthropicBackend:
    """Claude via the Anthropic Messages API with structured JSON output.

    Uses the server-side refusal fallback (``fallbacks="default"``); a final
    refusal or a truncated response is surfaced as :class:`BackendError` and
    becomes ``insufficient_evidence``.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        client: Any = None,
        effort: str = "high",
        max_tokens: int = 16000,
    ) -> None:
        """``client`` defaults to ``anthropic.Anthropic()`` (reads ANTHROPIC_API_KEY)."""
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def complete(self, incident: Incident, system: str, user: str) -> str:
        """Call the API and return the JSON text block."""
        import anthropic

        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
                },
            )
        except anthropic.APIStatusError as exc:
            raise BackendError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise BackendError(f"Anthropic API connection error: {exc}") from exc
        if resp.stop_reason == "refusal":
            raise BackendError("model declined the request (refusal)")
        if resp.stop_reason == "max_tokens":
            raise BackendError("model output truncated (max_tokens)")
        text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), None)
        if text is None:
            raise BackendError("no text block in response")
        return str(text)


def _fixture_path(fixtures_dir: Path, incident_id: str) -> Path:
    safe = re.sub(r"[^\w.-]", "_", incident_id)
    return fixtures_dir / f"{safe}.json"


class ReplayBackend:
    """Replays recorded (or hand-authored) responses keyed by incident ID.

    Fixtures record the prompt hash they were produced for; a mismatch is
    reported in :attr:`stale` so prompt changes cannot silently reuse old
    answers.
    """

    name = "replay"

    def __init__(self, fixtures_dir: Path) -> None:
        """Load fixtures lazily from ``fixtures_dir``."""
        self.fixtures_dir = fixtures_dir
        self.stale: list[str] = []

    def complete(self, incident: Incident, system: str, user: str) -> str:
        """Return the stored response text."""
        path = _fixture_path(self.fixtures_dir, incident.id)
        if not path.exists():
            raise BackendError(f"no replay fixture for incident {incident.id} ({path.name})")
        fx = json.loads(path.read_text())
        if fx.get("prompt_sha") != prompt_sha(system, user):
            self.stale.append(incident.id)
        return json.dumps(fx["response"])


class RecordingBackend:
    """Wraps a live backend and saves each response as a replay fixture."""

    def __init__(self, inner: Backend, fixtures_dir: Path) -> None:
        """Record into ``fixtures_dir``."""
        self.inner = inner
        self.fixtures_dir = fixtures_dir
        self.name = f"{inner.name}+record"

    def complete(self, incident: Incident, system: str, user: str) -> str:
        """Call the inner backend and persist the result."""
        raw = self.inner.complete(incident, system, user)
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        fx = {
            "incident_id": incident.id,
            "provenance": f"recorded from {self.inner.name} backend",
            "prompt_sha": prompt_sha(system, user),
            "evidence_sha": incident.evidence_sha(),
            "response": json.loads(raw),
        }
        _fixture_path(self.fixtures_dir, incident.id).write_text(json.dumps(fx, indent=2) + "\n")
        return raw
