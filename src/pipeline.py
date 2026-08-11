"""
Technical review pipeline: uniform check results and gate aggregation.

Extracted from notebooks/16-integrated-pipeline.ipynb.

The TechnicalReviewRunner runs every registered technical check (solar
access, view corridor, sponge city, setback, living circle) and
aggregates them into a single AND-gate decision for haidian's
procedure.py TechnicalAnalysis phase:

- a check that FAILS with severity CRITICAL or MAJOR blocks the gate;
- a NOT_ASSESSED check (data-gated: official data missing) never blocks
  the gate, but is reported separately so the gate never silently
  certifies a check that did not actually run;
- checks receive only the kwargs their own signature declares
  (``inspect.signature`` filtering, never blind forwarding).

Standard references per check
-----------------------------
- check_solar: GB 50180-2018 表 4.0.9 — Beijing 大寒日 >= 2 h for general
  housing; the D/H >= 1.6 coefficient is 大寒日-based practice.
- check_view_corridor: 北京城市总体规划 (2016-2035) 第52条, DB11/T 1945-2021.
- check_sponge_city: DB11/685-2021 — Beijing 新建项目年径流总量控制率
  >= 85% (design rainfall H = 33.6 mm).
- check_setback: DB11/T 996-2013, GB 50016-2014 fire separation.
- check_living_circle: TD/T 1062-2021 facility configuration rates.

All checks are deterministic CODE steps — zero GPU, zero LLM.
"""

import inspect
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

try:
    from .sponge import design_rainfall_mm
except ImportError:
    from sponge import design_rainfall_mm

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Severity(Enum):
    CRITICAL = 'critical'          # Blocks gate, must fix
    MAJOR = 'major'                # Blocks gate
    MINOR = 'minor'                # Advisory only
    INFO = 'info'                  # Informational
    NOT_ASSESSED = 'not_assessed'  # Data-gated: cannot verify (official data missing)


class Category(Enum):
    """Analysis categories from the notebook 10-14 classification."""

    COMPUTE_INTENSIVE = 'A'   # Simulation-based (solar, sponge)
    DATA_INTENSIVE = 'B'      # Data-gated (view corridor, three-lines)
    PARAMETER_SENSITIVE = 'C' # Parameter-dominated (setback, runoff coeff)


# ---------------------------------------------------------------------------
# CheckResult
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Uniform result from a technical analysis check.

    Attributes
    ----------
    check_id : str
        Stable identifier, e.g. 'CHECK-SOLAR-001'.
    check_name : str
        Human-readable name.
    passed : bool
        Whether the check passed.
    evidence : dict
        Machine-readable evidence trail for tech_review.json.
    severity : Severity, optional
        Default MAJOR.
    category : Category, optional
        Default COMPUTE_INTENSIVE.
    detail : str, optional
        Human-readable summary.
    error : str or None, optional
        Set when the check itself failed to execute.
    """

    check_id: str
    check_name: str
    passed: bool
    evidence: Dict[str, Any] = field(default_factory=dict)
    severity: Severity = Severity.MAJOR
    category: Category = Category.COMPUTE_INTENSIVE
    detail: str = ''
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize for tech_review.json."""
        return {
            'check_id': self.check_id,
            'name': self.check_name,
            'passed': self.passed,
            'severity': self.severity.value,
            'category': self.category.value,
            'detail': self.detail,
            'evidence': self.evidence,
            'error': self.error,
        }


# ---------------------------------------------------------------------------
# Individual technical checks
# ---------------------------------------------------------------------------


def check_solar(buildings: List[dict],
                site_lat: float = 39.9,
                d_h_standard: float = 1.6) -> CheckResult:
    """Check solar access for residential buildings (spacing coefficient).

    Simplified gate-level check: for buildings labeled 'residential',
    verify the produced ``spacing_to_south_m`` satisfies D/H >= 1.6.
    The standard coefficient is calibrated on 大寒日 analysis (GB
    50180-2018 表 4.0.9); a building without ``spacing_to_south_m``
    FAILS CLOSED — missing pipeline-produced data is a hard failure,
    never a silent pass.

    Parameters
    ----------
    buildings : list of dict
        GeoJSON-style features with ``properties``.
    site_lat : float, optional
        Site latitude (default Beijing).
    d_h_standard : float, optional
        Minimum D/H coefficient (default 1.6, Beijing 大寒日 practice).

    Returns
    -------
    CheckResult
    """
    residential = [b for b in buildings
                   if b.get('properties', {}).get('building_type') == 'residential']
    if not residential:
        return CheckResult(
            'CHECK-SOLAR-001', 'Solar Access (Spacing Coefficient)',
            passed=True, evidence={'residential_count': 0},
            severity=Severity.INFO,
            detail='No residential buildings to check',
            category=Category.COMPUTE_INTENSIVE)

    violations = []
    for b in residential:
        props = b.get('properties', {})
        h = props.get('height_m', 18)
        # NOTE: no default fallback.  spacing_to_south_m must be produced
        # by the generation pipeline; a building without it FAILS CLOSED.
        spacing = props.get('spacing_to_south_m')
        required = h * d_h_standard
        if spacing is None:
            violations.append({
                'building_id': props.get('id'),
                'height': h, 'spacing': None, 'required': required,
                'reason': 'missing_spacing_to_south_m',
            })
        elif spacing < required:
            violations.append({
                'building_id': props.get('id'),
                'height': h, 'spacing': spacing, 'required': required,
            })

    missing_data = [v for v in violations
                    if v.get('reason') == 'missing_spacing_to_south_m']
    confirmed = len(violations) - len(missing_data)
    passed = len(violations) == 0
    return CheckResult(
        'CHECK-SOLAR-001', 'Solar Access (Spacing Coefficient)',
        passed=passed,
        evidence={
            'total_residential': len(residential),
            'violations': confirmed,
            'missing_spacing_data': len(missing_data),
            'violation_details': violations,
        },
        severity=Severity.CRITICAL if not passed else Severity.INFO,
        detail=(
            f'{len(missing_data)}/{len(residential)} buildings missing spacing_to_south_m (cannot verify); '
            f'{confirmed} confirmed violations'
            if violations else 'All residential buildings pass'),
        category=Category.COMPUTE_INTENSIVE)


def check_view_corridor(buildings: List[dict],
                        corridors_defined: bool = False) -> CheckResult:
    """Check building heights against view corridor constraints.

    Data-gated: without the official corridor GIS the constraint cannot
    be verified and the check is reported honestly as NOT_ASSESSED — the
    gate must NEVER silently certify compliance for a check that did not
    run.

    Parameters
    ----------
    buildings : list of dict
        GeoJSON-style features with ``properties`` (``height_m`` and
        ``view_corridor_max_height_m``).
    corridors_defined : bool, optional
        Whether official corridor data is available.

    Returns
    -------
    CheckResult
    """
    if not corridors_defined:
        return CheckResult(
            'CHECK-VIEW-001', 'View Corridor Compliance',
            passed=False,
            evidence={'corridors_available': False},
            severity=Severity.NOT_ASSESSED,
            detail='View corridors not defined — NOT ASSESSED (official GIS unavailable)',
            category=Category.DATA_INTENSIVE)

    violations = []
    for b in buildings:
        max_allowed = b.get('properties', {}).get('view_corridor_max_height_m', 999)
        actual = b.get('properties', {}).get('height_m', 0)
        if actual > max_allowed:
            violations.append({
                'building_id': b.get('properties', {}).get('id'),
                'actual_height': actual, 'max_allowed': max_allowed,
            })

    passed = len(violations) == 0
    return CheckResult(
        'CHECK-VIEW-001', 'View Corridor Compliance',
        passed=passed,
        evidence={'violations': len(violations), 'details': violations},
        severity=Severity.CRITICAL if not passed else Severity.INFO,
        detail=(f'{len(violations)} building(s) exceed view corridor height limit'
                if violations else 'All buildings within view corridor limits'),
        category=Category.DATA_INTENSIVE)


def check_sponge_city(land_use_features: List[dict],
                      site_area_sqm: float = 11400000.0,
                      target_control_rate: float = 0.85) -> CheckResult:
    """Check if LID provisions meet the sponge-city volume requirement.

    Beijing new-development standard: 年径流总量控制率 α >= 85%
    (DB11/685-2021), design rainfall depth H = 33.6 mm (容积法,
    GB 50014-2021).  The older 75% rate (DB11/685-2013) no longer
    governs new projects.

    Parameters
    ----------
    land_use_features : list of dict
        Features with ``properties.land_use_type`` and ``area_sqm``.
    site_area_sqm : float, optional
        Fallback site area when feature areas are absent.
    target_control_rate : float, optional
        Governing control rate (default 0.85 per DB11/685-2021).

    Returns
    -------
    CheckResult
    """
    H_mm = design_rainfall_mm(target_control_rate)
    # Runoff coefficients per land-use type
    phi_map = {'residential': 0.60, 'commercial': 0.70, 'industrial': 0.65,
               'green': 0.15, 'road': 0.85, 'public': 0.50, 'water': 1.0}

    total_area = 0.0
    weighted_phi = 0.0
    green_area = 0.0
    for f in land_use_features:
        props = f.get('properties', {})
        lu_type = props.get('land_use_type', 'residential')
        area = props.get('area_sqm', 0)
        phi = phi_map.get(lu_type, 0.60)
        total_area += area
        weighted_phi += area * phi
        if lu_type == 'green':
            green_area += area

    if total_area < 1:
        total_area = site_area_sqm
        weighted_phi = site_area_sqm * 0.53

    phi_weighted = weighted_phi / total_area
    V_required = 10 * H_mm * phi_weighted * total_area / 10000  # m3
    # Assume LID storage from green space: V = area * depth * porosity
    lid_storage = green_area * 0.2 * 0.35  # 20 cm depth, 35% porosity

    passed = lid_storage >= V_required * 0.8  # 80% threshold for early design
    return CheckResult(
        'CHECK-SPONGE-001', 'Sponge City Volume Compliance',
        passed=passed,
        evidence={
            'V_required_m3': round(V_required, 1),
            'lid_storage_m3': round(lid_storage, 1),
            'phi_weighted': round(phi_weighted, 3),
            'green_area_sqm': round(green_area, 1),
            'design_rainfall_mm': H_mm,
        },
        severity=Severity.MAJOR if not passed else Severity.INFO,
        detail=(f'V_req={V_required:.0f}m3, LID={lid_storage:.0f}m3 ('
                + ('PASS' if passed else 'FAIL') + ')'),
        category=Category.COMPUTE_INTENSIVE)


def check_setback(buildings: List[dict]) -> CheckResult:
    """Check building setbacks from roads and property lines.

    Required setback by adjacent road class (typical reference values,
    DB11/T 996-2013 practice): arterial 15 m, secondary 10 m, local 5 m.

    Parameters
    ----------
    buildings : list of dict
        Features with ``properties.min_setback_m`` and
        ``properties.adjacent_road_class``.

    Returns
    -------
    CheckResult
    """
    violations = []
    for b in buildings:
        min_setback = b.get('properties', {}).get('min_setback_m', 999)
        road_class = b.get('properties', {}).get('adjacent_road_class', 'local')
        required = {'arterial': 15, 'secondary': 10, 'local': 5}.get(road_class, 5)
        if min_setback < required:
            violations.append({
                'building_id': b.get('properties', {}).get('id'),
                'min_setback': min_setback, 'required': required,
                'road_class': road_class,
            })

    passed = len(violations) == 0
    return CheckResult(
        'CHECK-SETBACK-001', 'Building Setback Compliance',
        passed=passed,
        evidence={'violations': len(violations), 'details': violations},
        severity=Severity.CRITICAL if not passed else Severity.INFO,
        detail=(f'{len(violations)} building(s) violate setback requirements'
                if violations else 'All setbacks compliant'),
        category=Category.PARAMETER_SENSITIVE)


def check_living_circle(facility_counts: Dict[str, int],
                        residential_pop: int = 0) -> CheckResult:
    """Check 15-minute living circle facility coverage.

    Facility rates per 1000 residents follow TD/T 1062-2021 basic-ensured
    (基础保障型) configuration guidance.

    Parameters
    ----------
    facility_counts : dict
        {facility_type: count}.
    residential_pop : int, optional
        Residential population served; 0 skips the check.

    Returns
    -------
    CheckResult
    """
    if residential_pop == 0:
        return CheckResult(
            'CHECK-LIVING-001', '15-Minute Living Circle',
            passed=True, evidence={'residential_pop': 0},
            severity=Severity.INFO,
            detail='No residential population to check',
            category=Category.COMPUTE_INTENSIVE)

    # Required facilities per 1000 residents (TD/T 1062-2021 guideline)
    requirements = {
        'kindergarten': 0.5, 'primary_school': 0.3, 'clinic': 0.2,
        'convenience_store': 2.0, 'park': 0.8, 'sports': 0.3,
        'elderly_care': 0.15, 'bus_stop': 1.0,
    }
    missing = []
    for facility, rate in requirements.items():
        required_count = max(1, int(residential_pop / 1000 * rate))
        actual = facility_counts.get(facility, 0)
        if actual < required_count:
            missing.append({'facility': facility,
                            'required': required_count, 'actual': actual})

    passed = len(missing) == 0
    return CheckResult(
        'CHECK-LIVING-001', '15-Minute Living Circle',
        passed=passed,
        evidence={'residential_pop': residential_pop,
                  'missing_facilities': missing},
        severity=Severity.MAJOR if not passed else Severity.INFO,
        detail=(f'{len(missing)} facility types below minimum'
                if missing else 'All facilities meet minimum'),
        category=Category.COMPUTE_INTENSIVE)


# ---------------------------------------------------------------------------
# TechnicalReviewRunner
# ---------------------------------------------------------------------------


class TechnicalReviewRunner:
    """Run all technical checks and aggregate results into a gate decision."""

    def __init__(self) -> None:
        self.checks: List[Callable] = []

    def register(self, check_fn: Callable) -> None:
        """Register a check function returning a :class:`CheckResult`."""
        self.checks.append(check_fn)

    def run_all(self, **kwargs) -> dict:
        """Run all registered checks.  Returns gate decision + evidence.

        Each check receives ONLY the kwargs its own signature declares —
        extra kwargs (e.g. ``land_use_features`` for ``check_solar``)
        are filtered via ``inspect.signature`` instead of crashing the
        check.  A check that raises is recorded as a CRITICAL error
        result, never swallowed silently.

        Returns
        -------
        dict
            Keys ``gate_passed``, ``total_checks``, ``passed``,
            ``failed``, ``not_assessed``, ``blocking_failures``,
            ``not_assessed_checks``, ``results``.
        """
        results = []
        for check_fn in self.checks:
            try:
                sig = inspect.signature(check_fn)
                check_kwargs = {k: v for k, v in kwargs.items()
                                if k in sig.parameters}
                result = check_fn(**check_kwargs)
                results.append(result)
            except Exception as e:  # defensive: never crash the gate
                results.append(CheckResult(
                    check_id='CHECK-ERROR',
                    check_name=check_fn.__name__,
                    passed=False,
                    evidence={},
                    severity=Severity.CRITICAL,
                    error=str(e),
                    detail=f'Check execution failed: {e}',
                ))

        # Gate decision: only CRITICAL/MAJOR failures block.  NOT_ASSESSED
        # (data-gated) checks never block — reported separately, so the
        # gate never silently certifies a check that did not run.
        blocking = [r for r in results
                    if r.severity in (Severity.CRITICAL, Severity.MAJOR)
                    and not r.passed]
        not_assessed = [r for r in results
                        if r.severity == Severity.NOT_ASSESSED]
        gate_passed = len(blocking) == 0

        return {
            'gate_passed': gate_passed,
            'total_checks': len(results),
            'passed': sum(1 for r in results if r.passed),
            'failed': sum(1 for r in results
                          if not r.passed and r.severity != Severity.NOT_ASSESSED),
            'not_assessed': len(not_assessed),
            'blocking_failures': len(blocking),
            'not_assessed_checks': [r.to_dict() for r in not_assessed],
            'results': [r.to_dict() for r in results],
        }

    def save_report(self, path: str, **kwargs) -> dict:
        """Run all checks and persist the aggregated report as JSON.

        The report is the same dict :meth:`run_all` returns (gate
        decision + per-check evidence trail), written to ``path`` with
        ``ensure_ascii=False`` so Chinese detail strings survive intact.

        Parameters
        ----------
        path : str
            Output file path for ``tech_review.json``.
        **kwargs
            Forwarded verbatim to :meth:`run_all` (each check receives
            only the kwargs its own signature declares).

        Returns
        -------
        dict
            The report dict that was written (identical to ``run_all``
            output).
        """
        report = self.run_all(**kwargs)
        with open(path, 'w') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return report


__all__ = [
    "Severity", "Category", "CheckResult",
    "check_solar", "check_view_corridor", "check_sponge_city",
    "check_setback", "check_living_circle", "TechnicalReviewRunner",
]
