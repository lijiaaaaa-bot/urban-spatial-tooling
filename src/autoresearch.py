"""
Autoresearch Lab — goal-driven exploration engine.

Extracted from haidian's goal-driven model (constraints/engine.py,
scripts/goal_driven_loop.py, docs/goal-driven-entry.md).

Core pattern (lidangzzz goal-driven):
  while (criteria not met):
      1. deterministic validation -> specific failures + evidence
      2. feed failures to subagent (feedback-as-prompt)
      3. stall detection -> continue or escalate
      4. subagent works, then repeat

Adapted for urban-spatial-tooling self-exploration:
  - "submission" = the project itself (src/, tests/, notebooks/)
  - "constraints" = exploration targets (D-01 must have facade_insolation, etc.)
  - "subagent" = external build agent (dispatched via prompt)
  - "criteria met" = all targets at active priority level done
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


# ═══════════════════════════════════════════════════════════════
# Exploration Target — the "constraint" to satisfy
# ═══════════════════════════════════════════════════════════════

class ExplorationAxis(Enum):
    DEPTH = "depth"
    BREADTH = "breadth"

class AnalysisType(Enum):
    A = "A"  # Compute-intensive (simulation)
    B = "B"  # Data-intensive (data-gated)
    C = "C"  # Parameter-sensitive

class TargetOutcome(Enum):
    PASS = "PASS"        # Target satisfied
    FAIL = "FAIL"        # Target not done
    SKIP = "SKIP"        # Data-gated, cannot verify
    IN_PROGRESS = "IN_PROGRESS"  # Agent working on it
    ERROR = "ERROR"      # Check failed to execute

@dataclass
class TargetResult:
    """One exploration target check result — mirrors haidian ConstraintResult."""

    target_id: str
    name: str
    outcome: TargetOutcome
    priority: int
    analysis_type: AnalysisType
    axis: ExplorationAxis
    evidence: str = ""        # Concrete file:line or "function X not in module Y"
    detail: str = ""          # Human-readable summary
    elapsed_ms: float = 0.0

    @property
    def is_pass(self) -> bool:
        return self.outcome == TargetOutcome.PASS

    @property
    def is_blocking(self) -> bool:
        """P1 targets that fail are blocking."""
        return self.priority == 1 and self.outcome == TargetOutcome.FAIL

    @property
    def fingerprint(self) -> str:
        """Stable fingerprint for stall detection."""
        token = f"{self.target_id}:{self.evidence}"
        return hashlib.sha256(token.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════
# Exploration Space — the "constraint registry"
# ═══════════════════════════════════════════════════════════════

EXPLORATION_SPACE: list[dict] = [
    # ── DEPTH ──
    {"id": "D-01", "name": "Per-window cumulative insolation (窗台面累计日照)",
     "axis": ExplorationAxis.DEPTH, "type": AnalysisType.A, "priority": 1,
     "check": ("src/solar.py", "facade_insolation"),
     "description": "Extend nb10: ground grid → building facade window-level 2h check per GB 50180-2018"},
    {"id": "D-02", "name": "Continuous sponge simulation (SWMM-style)",
     "axis": ExplorationAxis.DEPTH, "type": AnalysisType.A, "priority": 2,
     "check": ("src/sponge.py", "continuous_simulation"),
     "description": "Extend nb12: static volume method → continuous rainfall-runoff"},
    {"id": "D-03", "name": "Multi-corridor 3D building envelope",
     "axis": ExplorationAxis.DEPTH, "type": AnalysisType.B, "priority": 2,
     "check": ("src/compliance.py", "combine_corridors"),
     "description": "Extend nb11: single fan → multi-corridor 3D height surface + terrain"},
    {"id": "D-04", "name": "Multi-modal living circle (transit + bike + walk)",
     "axis": ExplorationAxis.DEPTH, "type": AnalysisType.A, "priority": 2,
     "check": ("src/living_circle.py", "multi_modal_score"),
     "description": "Extend nb13: walking only → cycling + bus + combined accessibility"},
    {"id": "D-05", "name": "Fire-separation between buildings",
     "axis": ExplorationAxis.DEPTH, "type": AnalysisType.C, "priority": 1,
     "check": ("src/setback.py", "check_fire_separation"),
     "description": "Extend nb14: road setbacks → inter-building fire separation GB 50016-2014"},
    {"id": "D-06", "name": "Building solar irradiation on facades (annual)",
     "axis": ExplorationAxis.DEPTH, "type": AnalysisType.A, "priority": 3,
     "check": ("src/solar.py", "facade_annual_irradiation"),
     "description": "Extend nb10: winter solstice only → annual cumulative facade irradiation"},

    # ── BREADTH ──
    {"id": "B-01", "name": "Ventilation / wind corridor analysis (通风分析)",
     "axis": ExplorationAxis.BREADTH, "type": AnalysisType.A, "priority": 1,
     "check": ("src/ventilation.py", "ventilation_corridors"),
     "description": "Beijing mandatory. Wind-pressure model + frontal area density + ventilation corridors. nb17"},
    {"id": "B-02", "name": "Traffic capacity analysis (交通承载力)",
     "axis": ExplorationAxis.BREADTH, "type": AnalysisType.A, "priority": 2,
     "check": ("src/traffic.py", "road_saturation"),
     "description": "Road network capacity + saturation ratio. nb19. GB/T 51328-2018"},
    {"id": "B-03", "name": "Urban renewal classification (城市更新 留-改-拆)",
     "axis": ExplorationAxis.BREADTH, "type": AnalysisType.C, "priority": 1,
     "check": ("src/renewal.py", "classify_building"),
     "description": "Building-level retain/renovate/demolish. 北京市城市更新条例(2022). nb18"},
    {"id": "B-04", "name": "Population-facility-land closure (人口-设施-用地平衡)",
     "axis": ExplorationAxis.BREADTH, "type": AnalysisType.C, "priority": 2,
     "check": ("src/closure.py", "check_population_balance"),
     "description": "Population → facility demand → land allocation consistency. GB 50180-2018 附录A. nb20"},
    {"id": "B-05", "name": "Vertical design / grading (竖向设计)",
     "axis": ExplorationAxis.BREADTH, "type": AnalysisType.B, "priority": 3,
     "check": ("src/vertical.py", "cut_fill_balance"),
     "description": "Terrain slope/aspect, cut-fill. Needs DEM (data-gated). nb21"},
]


# ═══════════════════════════════════════════════════════════════
# Autoresearch Engine — deterministic constraint checker
# ═══════════════════════════════════════════════════════════════

class AutoresearchEngine:
    """Deterministic exploration target validator.

    Mirrors haidian's ConstraintEngine: loads a "registry" (EXPLORATION_SPACE),
    runs deterministic checks, returns structured results with fingerprints.
    Zero LLM. Zero GPU.
    """

    def __init__(self, repo_root: str | Path = "."):
        self.root = Path(repo_root)
        self.targets = EXPLORATION_SPACE

    def check_target(self, target: dict) -> TargetResult:
        """Check one exploration target. Deterministic.

        Depth target: PASS if check function exists in src module.
        Breadth target: PASS if notebook + module + test all exist.
        """
        t0 = time.time()
        tid = target["id"]
        module_rel, func_name = target["check"]

        module_path = self.root / module_rel

        # ── Module check ──
        if not module_path.exists():
            return TargetResult(
                target_id=tid, name=target["name"],
                outcome=TargetOutcome.FAIL, priority=target["priority"],
                analysis_type=target["type"], axis=target["axis"],
                evidence=f"{module_rel} does not exist",
                detail=f"Module not created yet",
                elapsed_ms=(time.time() - t0) * 1000)

        # ── Function check ──
        text = module_path.read_text()
        if f"def {func_name}" not in text:
            return TargetResult(
                target_id=tid, name=target["name"],
                outcome=TargetOutcome.FAIL, priority=target["priority"],
                analysis_type=target["type"], axis=target["axis"],
                evidence=f"{module_rel}: function '{func_name}' not found",
                detail=f"Function {func_name} missing from {module_rel}",
                elapsed_ms=(time.time() - t0) * 1000)

        # ── Test check (depth targets must have tests) ──
        if target["axis"] == ExplorationAxis.DEPTH:
            module_stem = Path(module_rel).stem
            test_file = self.root / "tests" / f"test_{module_stem}.py"
            if test_file.exists():
                test_text = test_file.read_text()
                if func_name not in test_text and target["id"] not in test_text:
                    return TargetResult(
                        target_id=tid, name=target["name"],
                        outcome=TargetOutcome.FAIL, priority=target["priority"],
                        analysis_type=target["type"], axis=target["axis"],
                        evidence=f"tests/test_{module_stem}.py: no test for '{func_name}'",
                        detail=f"Function extracted but no test coverage",
                        elapsed_ms=(time.time() - t0) * 1000)

        # ── Notebook/content check (any notebook referencing this target) ──
        if target["axis"] == ExplorationAxis.BREADTH:
            all_notebooks = list(self.root.glob("notebooks/*.ipynb"))
            found = False
            for nb in all_notebooks:
                try:
                    nb_text = nb.read_text()
                    if tid in nb_text or func_name in nb_text:
                        found = True
                        break
                except Exception:
                    pass
            if not found:
                return TargetResult(
                    target_id=tid, name=target["name"],
                    outcome=TargetOutcome.FAIL, priority=target["priority"],
                    analysis_type=target["type"], axis=target["axis"],
                    evidence=f"No notebook references target {tid} or function {func_name}",
                    detail=f"Notebook not created yet",
                    elapsed_ms=(time.time() - t0) * 1000)

        return TargetResult(
            target_id=tid, name=target["name"],
            outcome=TargetOutcome.PASS, priority=target["priority"],
            analysis_type=target["type"], axis=target["axis"],
            evidence=f"{module_rel}:{func_name}() + tests + notebook",
            detail=f"Target satisfied",
            elapsed_ms=(time.time() - t0) * 1000)

    def validate(self, priority_filter: int | None = None) -> list[TargetResult]:
        """Run all target checks. Returns list of TargetResult with fingerprints.

        Mirrors ConstraintEngine.validate().
        """
        results = []
        for target in self.targets:
            if priority_filter is not None and target["priority"] != priority_filter:
                continue
            results.append(self.check_target(target))
        return results

    def report(self, results: list[TargetResult]) -> dict:
        """Aggregate results into a gate decision. Mirrors ConstraintEngine.report().

        Returns:
            {"criteria_met": bool, "blocking_failures": int,
             "passed": int, "failed": int, "results": [...],
             "fingerprints": set, "ready_for_dispatch": [...target_ids...]}
        """
        blocking = [r for r in results if r.is_blocking]
        all_fail = [r for r in results if r.outcome == TargetOutcome.FAIL]
        fingerprints = {r.fingerprint for r in all_fail}

        return {
            "criteria_met": len(blocking) == 0,
            "blocking_failures": len(blocking),
            "passed": sum(1 for r in results if r.is_pass),
            "failed": len(all_fail),
            "total": len(results),
            "fingerprints": fingerprints,
            "ready_for_dispatch": [r.target_id for r in all_fail],
        }


# ═══════════════════════════════════════════════════════════════
# Stall Detector — mirrors haidian ConstraintEngine.stall_check()
# ═══════════════════════════════════════════════════════════════

@dataclass
class StallDetector:
    """Detect when the loop is stuck — same fingerprints across rounds."""

    threshold: int = 2
    history: list[set] = field(default_factory=list)
    round_num: int = 0

    def check(self, fingerprints: set) -> bool:
        """Returns True if STALL detected (same fingerprints consecutively)."""
        self.round_num += 1
        self.history.append(fingerprints)
        if len(self.history) < self.threshold:
            return False
        recent = self.history[-self.threshold:]
        return all(fp == fingerprints for fp in recent)

    def escalate(self, reason: str) -> dict:
        """Escalate to human — stall detected."""
        return {
            "status": "escalated",
            "reason": reason,
            "round": self.round_num,
            "history": [sorted(list(h)) for h in self.history],
        }


# ═══════════════════════════════════════════════════════════════
# Goal-Driven Master Loop
# ═══════════════════════════════════════════════════════════════

class GoalDrivenExplorer:
    """Master loop for autonomous exploration.

    Pattern (from haidian goal_driven_loop.py):
      while (criteria not met):
          validate -> feed failures -> stall check -> dispatch -> wait -> repeat

    The Master has exactly 3 jobs:
      1. Run deterministic validation (AutoresearchEngine)
      2. Check if criteria are met (all priority-N targets PASS)
      3. Dispatch subagent to fix failures, or escalate

    The Master does NOT:
      - Tell the subagent HOW to fix (feedback-as-prompt: "target X: evidence Y")
      - Distinguish "generate" from "fix" (a missing target is just a FAIL)
      - Use LLM (all validation is deterministic CODE)
    """

    def __init__(self, repo_root: str | Path = "."):
        self.root = Path(repo_root)
        self.engine = AutoresearchEngine(repo_root)
        self.stall = StallDetector(threshold=2)
        self.max_rounds = 10
        self.active_priority = 1  # Start with P1

    def run(self, max_priority: int = 2) -> dict:
        """Run the goal-driven loop for exploration targets.

        Args:
            max_priority: Maximum priority level to enforce (1=P1 only, 2=P1+P2, 3=all)

        Returns:
            {"status": "completed" | "escalated" | "max_rounds",
             "rounds": int, "results": [...], "escalation": dict | None}
        """
        round_num = 0

        while round_num < self.max_rounds:
            round_num += 1

            # ── 1. Validate all targets at active priority ──
            results = []
            for p in range(1, self.active_priority + 1):
                results.extend(self.engine.validate(priority_filter=p))

            report = self.engine.report(results)

            # ── 2. Criteria met? ──
            if report["criteria_met"]:
                # Advance to next priority level
                if self.active_priority < max_priority:
                    self.active_priority += 1
                    self.stall = StallDetector(threshold=2)  # Reset stall for new priority
                    continue
                return {
                    "status": "completed",
                    "rounds": round_num,
                    "report": report,
                    "escalation": None,
                }

            # ── 3. Stall detection ──
            fingerprints = report["fingerprints"]
            if self.stall.check(fingerprints):
                return {
                    "status": "escalated",
                    "rounds": round_num,
                    "report": report,
                    "escalation": self.stall.escalate(
                        f"Stall: same {len(fingerprints)} failure fingerprints "
                        f"for {self.stall.threshold} consecutive rounds at priority {self.active_priority}"
                    ),
                }

            # ── 4. Dispatch — produce structured failure prompts ──
            failures = [r for r in results if r.outcome == TargetOutcome.FAIL]
            dispatch = self._build_dispatch(failures, round_num)

            # In a real loop, this is where we'd call the subagent.
            # For now, return the dispatch so an external agent can pick it up.
            return {
                "status": "awaiting_subagent",
                "rounds": round_num,
                "report": report,
                "dispatch": dispatch,
                "escalation": None,
            }

        return {
            "status": "max_rounds",
            "rounds": round_num,
            "report": report,
            "escalation": self.stall.escalate(f"Max rounds ({self.max_rounds}) exceeded"),
        }

    def _build_dispatch(self, failures: list[TargetResult], round_num: int) -> dict:
        """Build structured dispatch prompts for subagent(s).

        Feedback-as-prompt pattern: each failure becomes a "fix this" instruction
        with specific evidence, never "you are in phase N, generate X".
        """
        prompts = []
        for r in failures:
            prompts.append({
                "target_id": r.target_id,
                "name": r.name,
                "priority": r.priority,
                "type": r.analysis_type.value,
                "axis": r.axis.value,
                "prompt": (
                    f"Fix target {r.target_id}: {r.name}\n"
                    f"Evidence: {r.evidence}\n"
                    f"Detail: {r.detail}\n"
                    f"Priority: P{r.priority} | Type: {r.analysis_type.value} | Axis: {r.axis.value}\n"
                    f"Round: {round_num}"
                ),
            })
        return {
            "round": round_num,
            "num_failures": len(failures),
            "priority": self.active_priority,
            "prompts": prompts,
        }

    def continue_after_subagent(self, previous_result: dict) -> dict:
        """Resume the loop after a subagent completed work.

        Call this with the previous run() result after the subagent finishes.
        """
        return self.run()


# ═══════════════════════════════════════════════════════════════
# Quick inspection (not the loop — just a status snapshot)
# ═══════════════════════════════════════════════════════════════

def status(repo_root: str | Path = ".") -> str:
    """One-shot status report. No loop. For human inspection."""
    root = Path(repo_root)
    engine = AutoresearchEngine(root)

    lines = [
        "=" * 60,
        "AUTORESEARCH LAB — Goal-Driven Status",
        "=" * 60,
        "",
        f"Project: {root.resolve().name}",
    ]

    # Count tests
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q"],
            capture_output=True, text=True, cwd=str(root), timeout=60)
        for line in result.stdout.splitlines() + result.stderr.splitlines():
            if "passed" in line:
                lines.append(f"Tests: {line.strip()}")
                break
    except Exception:
        lines.append("Tests: (could not run)")

    # Count modules
    modules = list(root.glob("src/*.py"))
    lines.append(f"Modules: {len(modules)}")

    # Validate all targets
    all_results = engine.validate()
    report = engine.report(all_results)

    lines.extend([
        "",
        f"Targets: {report['total']} total | {report['passed']} PASS | {report['failed']} FAIL | {report['blocking_failures']} blocking",
        "",
    ])

    # By priority
    for p in [1, 2, 3]:
        p_results = [r for r in all_results if r.priority == p]
        p_pass = sum(1 for r in p_results if r.is_pass)
        p_fail = sum(1 for r in p_results if r.outcome == TargetOutcome.FAIL)
        if p_results:
            lines.append(f"  P{p}: {p_pass}/{len(p_results)} done ({p_fail} open)")

    lines.append("")
    lines.append(f"Criteria met (P1+): {report['criteria_met']}")
    lines.append(f"Ready for dispatch: {report['ready_for_dispatch']}")

    # Show failures with evidence
    failures = [r for r in all_results if r.outcome == TargetOutcome.FAIL]
    if failures:
        lines.append("")
        lines.append("--- Failures (feedback-as-prompt) ---")
        for r in sorted(failures, key=lambda x: (x.priority, x.target_id)):
            lines.append(f"  {r.target_id} [P{r.priority}] {r.name}")
            lines.append(f"    {r.evidence}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    print(status(repo))

    # Run the goal-driven loop in dry-run mode (will return after first validate)
    print("\n" + "=" * 60)
    print("Goal-Driven Loop (dry run — stops at dispatch)")
    print("=" * 60)
    explorer = GoalDrivenExplorer(repo)
    result = explorer.run(max_priority=2)
    print(f"Status: {result['status']}")
    print(f"Rounds: {result['rounds']}")
    report = result["report"]
    print(f"Criteria met: {report['criteria_met']}")
    print(f"Blocking failures: {report['blocking_failures']}")
    if result.get("dispatch"):
        d = result["dispatch"]
        print(f"Dispatch: {d['num_failures']} failures at priority {d['priority']}")
        for p in d["prompts"]:
            print(f"  → {p['target_id']}: {p['prompt'][:80]}...")
