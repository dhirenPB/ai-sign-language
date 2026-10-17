"""
Test registry and final readiness decision. Pure Python (no numpy / torch).

A check only counts as PASS when its block ran to the end without an exception. A block that
cannot be verified in the current environment (for example CUDA tests on a CPU runtime) must call
`chk.unverified("reason")`, which is recorded as UNVERIFIED and blocks the READY decision.
"""
import json
import traceback
from collections import OrderedDict

PASS, FAIL, UNVERIFIED = "PASS", "FAIL", "UNVERIFIED"


class Checks:
    def __init__(self):
        self.results = OrderedDict()

    def record(self, name, status, detail=""):
        self.results[name] = {"status": status, "detail": str(detail)}

    def status(self, name):
        return self.results.get(name, {}).get("status", UNVERIFIED)

    def all_passed(self, names):
        return all(self.status(n) == PASS for n in names)

    def __call__(self, name):
        return _Check(self, name)

    def table(self, names=None):
        names = list(self.results) if names is None else names
        rows = []
        for n in names:
            r = self.results.get(n, {"status": UNVERIFIED, "detail": "not run"})
            rows.append((r["status"], n, r["detail"]))
        return rows

    def print_table(self, names=None):
        for status, name, detail in self.table(names):
            print(f"[{status:10s}] {name}" + (f"  --  {detail}" if detail else ""))


class _Check:
    def __init__(self, registry, name):
        self.registry, self.name = registry, name
        self.detail = ""
        self._unverified = None

    def unverified(self, reason):
        self._unverified = reason

    def note(self, text):
        self.detail = text

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            if self._unverified:
                self.registry.record(self.name, UNVERIFIED, self._unverified)
            else:
                self.registry.record(self.name, PASS, self.detail)
        elif issubclass(exc_type, AssertionError):
            self.registry.record(self.name, FAIL, str(exc) or "assertion failed")
            print(f"[FAIL] {self.name}: {exc}")
        else:
            self.registry.record(self.name, FAIL, f"{exc_type.__name__}: {exc}")
            print(f"[FAIL] {self.name}: {exc_type.__name__}: {exc}")
            traceback.print_exception(exc_type, exc, tb)
        return True   # keep the notebook going; the final table shows every failure


def decide(checks, required):
    """READY only if every required check is PASS."""
    blockers = []
    for name in required:
        r = checks.results.get(name, {"status": UNVERIFIED, "detail": "not run"})
        if r["status"] != PASS:
            blockers.append({"check": name, "status": r["status"], "detail": r["detail"]})
    return ("READY FOR CONTROLLED TRAINING" if not blockers else "NOT READY"), blockers


def render_markdown(title, decision, blockers, checks, sections):
    lines = [f"# {title}", "", f"## Decision: {decision}", ""]
    if blockers:
        lines += ["Blockers:", ""]
        lines += [f"* **{b['check']}** is {b['status']}" + (f": {b['detail']}" if b["detail"] else "") for b in blockers]
        lines.append("")
    lines += ["## Test report", "", "| Status | Check | Detail |", "|---|---|---|"]
    for status, name, detail in checks.table():
        lines.append(f"| {status} | {name} | {str(detail).replace('|', '/')} |")
    for heading, body in sections.items():
        lines += ["", f"## {heading}", "", body if isinstance(body, str) else "```json\n" + json.dumps(body, indent=2) + "\n```"]
    return "\n".join(lines) + "\n"
