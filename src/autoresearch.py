"""
Autoresearch Lab — self-driving exploration engine.

Uses the project's own methodology to drive deeper and broader exploration:
- Reads current project state (CLAUDE.md, src/, tests/, notebooks/)
- Classifies the exploration space using the A/B/C framework
- Identifies gaps and deepening opportunities
- Dispatches experiments (notebook → module → test → document)

Principles (learned from this project):
1. Compare multiple approaches per problem, quantify error bounds
2. Classify analyses: A (compute-intensive) / B (data-intensive) / C (parameter-sensitive)
3. CPU-first. GPU only where proven (archived negative result).
4. Every experiment produces: notebook + src module + pytest + CLAUDE.md update
5. Gate semantics: fail-closed for missing data, NOT_ASSESSED for data-gated checks
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

# ── Exploration space ──────────────────────────────────────────────

class ExplorationAxis(Enum):
    DEPTH = "depth"    # Deeper on existing analysis
    BREADTH = "breadth"  # New analysis type

class AnalysisType(Enum):
    A = "A"  # Compute-intensive (simulation)
    B = "B"  # Data-intensive (data-gated)
    C = "C"  # Parameter-sensitive

@dataclass
class ExplorationTarget:
    """One potential exploration target."""
    id: str
    name: str
    axis: ExplorationAxis
    analysis_type: AnalysisType
    description: str
    parent_notebook: Optional[int] = None  # for depth: which nb to extend
    status: str = "open"  # open | in_progress | done
    priority: int = 1    # 1 (high) - 3 (low)

# ── The exploration space ──────────────────────────────────────────

EXPLORATION_SPACE: list[ExplorationTarget] = [
    # ── DEPTH: deepen existing analyses ──
    ExplorationTarget("D-01", "Per-window cumulative insolation (窗台面累计日照)",
        ExplorationAxis.DEPTH, AnalysisType.A,
        "Extend nb10: 地面栅格 → 建筑立面窗台面, 有效日照时间带 8:00-16:00, "
        "per-window 2h check per GB 50180-2018. Current: ground grid only.",
        parent_notebook=10, priority=1),
    ExplorationTarget("D-02", "Continuous sponge simulation (SWMM-style)",
        ExplorationAxis.DEPTH, AnalysisType.A,
        "Extend nb12: 容积法 → 连续降雨-径流模拟, design storm hydrograph, "
        "LID performance curves. Current: static volume method only.",
        parent_notebook=12, priority=2),
    ExplorationTarget("D-03", "3D building envelope from multiple view corridors",
        ExplorationAxis.DEPTH, AnalysisType.B,
        "Extend nb11: single fan → multi-corridor 3D height constraint surface, "
        "terrain-corrected (DEM). Current: single fan, flat ground.",
        parent_notebook=11, priority=2),
    ExplorationTarget("D-04", "Multi-modal living circle (transit + bike + walk)",
        ExplorationAxis.DEPTH, AnalysisType.A,
        "Extend nb13: walking-only → walking + cycling + bus, "
        "mode choice weights, time-based isochrones. Current: walking network only.",
        parent_notebook=13, priority=2),
    ExplorationTarget("D-05", "Fire-separation between buildings",
        ExplorationAxis.DEPTH, AnalysisType.C,
        "Extend nb14: road setbacks only → inter-building fire separation "
        "(GB 50016-2014 6/9/13m). Current: building-to-road only.",
        parent_notebook=14, priority=1),
    ExplorationTarget("D-06", "Building solar irradiation on facades",
        ExplorationAxis.DEPTH, AnalysisType.A,
        "Extend nb10: ground insolation → facade insolation (4 orientations), "
        "annual cumulative. Current: ground-only, winter solstice only.",
        parent_notebook=10, priority=3),

    # ── BREADTH: new analysis types ──
    ExplorationTarget("B-01", "Ventilation / wind corridor analysis (通风分析)",
        ExplorationAxis.BREADTH, AnalysisType.A,
        "Beijing mandatory (强条). Simplified CFD or empirical wind-pressure model, "
        "ventilation corridor identification, building height/wind speed relationship. "
        "New nb17. Reference: GB 50009-2012, 北京通风廊道规划.",
        priority=1),
    ExplorationTarget("B-02", "Traffic capacity analysis (交通承载力)",
        ExplorationAxis.BREADTH, AnalysisType.A,
        "Road network capacity, saturation ratio, intersection level-of-service. "
        "New nb18. Reference: GB/T 51328-2018, CJJ 37-2012.",
        priority=2),
    ExplorationTarget("B-03", "Urban renewal classification (城市更新 留-改-拆)",
        ExplorationAxis.BREADTH, AnalysisType.C,
        "Building-level retain/renovate/demolish classification based on "
        "age, structure, heritage status, land-use mismatch. "
        "New nb19. Reference: 北京市城市更新条例(2022).",
        priority=1),
    ExplorationTarget("B-04", "Population-facility-land closure (人口-设施-用地平衡)",
        ExplorationAxis.BREADTH, AnalysisType.C,
        "Verify population → facility demand → land allocation consistency. "
        "千人指标 cross-check. New nb20. Reference: GB 50180-2018 附录A.",
        priority=2),
    ExplorationTarget("B-05", "Vertical design / grading (竖向设计)",
        ExplorationAxis.BREADTH, AnalysisType.B,
        "Terrain slope/aspect analysis, cut-fill balance, drainage direction. "
        "Needs DEM data (data-gated). New nb21.",
        priority=3),
]

# ── State reader ───────────────────────────────────────────────────

def read_project_state(repo_root: str | Path = ".") -> dict:
    """Read the current project state from files on disk.

    Returns a dictionary with keys:
    - notebooks: list of notebook filenames
    - modules: list of src/*.py filenames
    - test_count: number of passing tests
    - test_modules: list of test filenames
    - metrics: outputs/metrics.json contents (or None)
    - goals_complete: goals marked [x] in CLAUDE.md
    - goals_pending: goals NOT marked [x] in CLAUDE.md
    - exploration_done: which EXPLORATION_SPACE targets have been done
    """
    root = Path(repo_root)

    state = {
        "notebooks": sorted(root.glob("notebooks/*.ipynb"), key=lambda p: p.name),
        "modules": sorted(root.glob("src/*.py")),
        "test_count": 0,
        "test_modules": sorted(root.glob("tests/test_*.py")),
        "metrics": None,
        "goals_complete": [],
        "goals_pending": [],
        "exploration_done": [],
    }

    # Read metrics
    metrics_path = root / "outputs" / "metrics.json"
    if metrics_path.exists():
        state["metrics"] = json.loads(metrics_path.read_text())

    # Count tests (run pytest once, parse output)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q"],
            capture_output=True, text=True, cwd=str(root), timeout=60)
        for line in result.stdout.splitlines() + result.stderr.splitlines():
            if "passed" in line:
                try:
                    state["test_count"] = int(line.split()[0])
                except (ValueError, IndexError):
                    pass
    except Exception:
        pass

    # Parse CLAUDE.md goals
    claude_md = root / "CLAUDE.md"
    if claude_md.exists():
        text = claude_md.read_text()
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("### [x]"):
                state["goals_complete"].append(line)
            elif line.startswith("### [ ]"):
                state["goals_pending"].append(line)

    # Check exploration space
    # Depth targets are "done" when their key function exists in src/.
    # Breadth targets are "done" when a new notebook with the target number exists.
    DEPTH_SIGNATURES = {
        "D-01": ("src/solar.py", "facade_insolation"),
        "D-02": ("src/sponge.py", "continuous_simulation"),
        "D-03": ("src/compliance.py", "multi_corridor_envelope"),
        "D-04": ("src/living_circle.py", "multi_modal_isochrone"),
        "D-05": ("src/setback.py", "check_fire_separation"),
        "D-06": ("src/solar.py", "facade_annual_irradiation"),
    }

    for target in EXPLORATION_SPACE:
        if target.axis == ExplorationAxis.DEPTH:
            sig = DEPTH_SIGNATURES.get(target.id)
            if sig:
                module_path, func_name = sig
                full_path = root / module_path
                if full_path.exists():
                    text = full_path.read_text()
                    if f"def {func_name}" in text:
                        state["exploration_done"].append(target.id)
        else:  # BREADTH
            nb_num = target.id.split("-")[1]
            expected_nb = f"{int(nb_num):02d}"
            matching = [n for n in state["notebooks"] if expected_nb in n.name]
            if matching:
                state["exploration_done"].append(target.id)

    return state


# ── Gap analyzer ───────────────────────────────────────────────────

def analyze_gaps(state: dict) -> dict:
    """Analyze exploration gaps from current state.

    Returns:
        {
            "depth_open": [...],    # depth targets not yet done
            "breadth_open": [...],  # breadth targets not yet done
            "total_open": int,
            "by_priority": {1: [...], 2: [...], 3: [...]},
            "by_type": {"A": [...], "B": [...], "C": [...]},
        }
    """
    done = set(state.get("exploration_done", []))

    depth_open = [t for t in EXPLORATION_SPACE
                  if t.axis == ExplorationAxis.DEPTH and t.id not in done]
    breadth_open = [t for t in EXPLORATION_SPACE
                    if t.axis == ExplorationAxis.BREADTH and t.id not in done]
    all_open = depth_open + breadth_open

    return {
        "depth_open": [t.id for t in depth_open],
        "breadth_open": [t.id for t in breadth_open],
        "total_open": len(depth_open) + len(breadth_open),
        "by_priority": {
            p: [t.id for t in all_open if t.priority == p]
            for p in [1, 2, 3]
        },
        "by_type": {
            typ: [t.id for t in all_open if t.analysis_type.value == typ]
            for typ in ["A", "B", "C"]
        },
        "detail": {t.id: {
            "name": t.name,
            "axis": t.axis.value,
            "type": t.analysis_type.value,
            "description": t.description,
            "priority": t.priority,
        } for t in all_open},
    }


# ── Experiment dispatcher ──────────────────────────────────────────

def dispatch_experiment(target_id: str, repo_root: str | Path = ".") -> dict:
    """Generate a prompt for building the next experiment.

    This does NOT execute the experiment — it produces the specification
    that an agent (or human) can execute. The spec includes:
    - Which notebook to create/extend
    - What methods to compare
    - What module to extract
    - What tests to add

    Returns a dict with the experiment specification.
    """
    target = None
    for t in EXPLORATION_SPACE:
        if t.id == target_id:
            target = t
            break

    if target is None:
        return {"error": f"Unknown target: {target_id}",
                "available": [t.id for t in EXPLORATION_SPACE]}

    root = Path(repo_root)

    # Determine notebook number
    if target.axis == ExplorationAxis.DEPTH:
        nb_num = target.parent_notebook
        nb_name = f"{nb_num:02d}" if nb_num else None
        is_extension = True
    else:
        existing = sorted(root.glob("notebooks/*.ipynb"))
        max_nb = 0
        for p in existing:
            try:
                n = int(p.name.split("-")[0])
                if n > max_nb:
                    max_nb = n
            except ValueError:
                pass
        nb_num = max_nb + 1
        nb_name = f"{nb_num:02d}"
        is_extension = False

    return {
        "target_id": target.id,
        "target_name": target.name,
        "axis": target.axis.value,
        "analysis_type": target.analysis_type.value,
        "priority": target.priority,
        "notebook": f"notebooks/{nb_name}-{target.name.split('(')[0].strip().lower().replace(' ', '-')[:40]}.ipynb"
        if not is_extension else f"notebooks/{nb_name}-*.ipynb (extend existing)",
        "is_extension": is_extension,
        "parent_notebook": target.parent_notebook,
        "description": target.description,
        "methodology": {
            "compare_approaches": f"Implement 2-3 methods for {target.name}, compare accuracy and performance",
            "classify": f"Type {target.analysis_type.value}: "
                       f"{'compute-intensive simulation' if target.analysis_type == AnalysisType.A else 'data-gated' if target.analysis_type == AnalysisType.B else 'parameter-sensitive'}",
            "extract_module": True,
            "add_tests": True,
            "update_claude_md": True,
        },
        "experiment_prompt": _build_experiment_prompt(target, nb_num, is_extension),
    }


def _build_experiment_prompt(target: ExplorationTarget,
                               nb_num: int,
                               is_extension: bool) -> str:
    """Build a prompt for executing this experiment."""
    parts = [
        f"Build exploration target {target.id}: {target.name}",
        f"",
        f"Type: {target.analysis_type.value} ({target.axis.value})",
        f"Priority: {target.priority}",
        f"",
        f"DESCRIPTION:",
        f"{target.description}",
        f"",
        f"REQUIREMENTS:",
    ]

    if is_extension:
        parts.append(f"- EXTEND existing notebook {target.parent_notebook:02d}")
        parts.append(f"- Add new cells for the deeper analysis")
    else:
        parts.append(f"- Create new notebook {nb_num:02d}")

    parts.extend([
        f"- Compare 2-3 methodological approaches",
        f"- Quantify error bounds between approaches",
        f"- Extract reusable functions to src/ module",
        f"- Add pytest tests (3-5 tests minimum)",
        f"- Save figures to outputs/ directory",
        f"- CPU only. No GPU. Use numpy + shapely.",
        f"- All cells must execute without errors.",
    ])

    if target.analysis_type == AnalysisType.B:
        parts.append("- Data-gated: if official data unavailable, use synthetic data and mark as provisional")

    return "\n".join(parts)


# ── Progress reporter ──────────────────────────────────────────────

def report(state: dict | None = None) -> str:
    """Generate a human-readable exploration progress report."""
    if state is None:
        state = read_project_state()

    gaps = analyze_gaps(state)

    lines = [
        "=" * 60,
        "AUTORESEARCH LAB — Exploration Status",
        "=" * 60,
        "",
        f"Project: urban-spatial-tooling",
        f"Modules: {len(state['modules'])}",
        f"Tests: {state['test_count']}",
        f"Goals complete: {len(state['goals_complete'])}",
        f"",
        f"Exploration space: {len(EXPLORATION_SPACE)} targets total",
        f"  Depth (deepen existing):  {len([t for t in EXPLORATION_SPACE if t.axis == ExplorationAxis.DEPTH])}",
        f"  Breadth (new analysis):   {len([t for t in EXPLORATION_SPACE if t.axis == ExplorationAxis.BREADTH])}",
        f"  Open: {gaps['total_open']}",
        f"",
        f"--- By Priority ---",
        f"  P1 (high):  {len(gaps['by_priority'][1])} open",
        f"  P2 (medium): {len(gaps['by_priority'][2])} open",
        f"  P3 (low):   {len(gaps['by_priority'][3])} open",
        f"",
        f"--- By Type ---",
        f"  A (compute-intensive): {len(gaps['by_type']['A'])} open",
        f"  B (data-intensive):    {len(gaps['by_type']['B'])} open",
        f"  C (parameter-sensitive): {len(gaps['by_type']['C'])} open",
        f"",
        f"--- P1 Targets (recommended next) ---",
    ]

    for t in EXPLORATION_SPACE:
        if t.priority == 1 and t.id in gaps['by_priority'][1]:
            lines.append(f"  {t.id}: {t.name} [{t.analysis_type.value}]")
            lines.append(f"    {t.description[:100]}...")

    lines.extend([
        "",
        "=" * 60,
        f"To dispatch: dispatch_experiment('TARGET-ID')",
        f"To see all: [t.id for t in EXPLORATION_SPACE]",
    ])

    return "\n".join(lines)


# ── Self-test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    repo = os.getcwd()
    if not (Path(repo) / "CLAUDE.md").exists():
        repo = os.path.dirname(__file__)
        if not (Path(repo) / "CLAUDE.md").exists():
            repo = "."

    state = read_project_state(repo)
    print(report(state))
