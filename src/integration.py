"""
End-to-end technical review for haidian submissions.

Wires the pieces together: a haidian submission's ``geometry/`` layer
files are loaded via :mod:`adapter`, adapted into the technical-review
pipeline's check inputs, run through :class:`TechnicalReviewRunner`
(the AND-gate from :mod:`pipeline`), and the full evidence report is
persisted as ``tech_review.json`` in the submission directory.

Flow (all deterministic CODE steps, zero GPU, zero external APIs):

    submission/geometry/*.geojson
        -> load_hadian_submission()        (adapter)
        -> adapt_building / adapt_land_use (adapter)
        -> TechnicalReviewRunner.run_all() (pipeline)
        -> tech_review.json                (report + meta)

The gate semantics follow :mod:`pipeline`: only CRITICAL/MAJOR failures
block the gate; NOT_ASSESSED (data-gated, e.g. view corridors without
official GIS) never blocks but is reported honestly.  Fields a haidian
submission cannot supply — official view-corridor data, facility
inventories — surface as NOT_ASSESSED / skipped checks rather than as
silent passes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from .adapter import (adapt_building, adapt_land_use,
                          load_hadian_submission)
    from .pipeline import (TechnicalReviewRunner, check_living_circle,
                           check_setback, check_solar, check_sponge_city,
                           check_view_corridor)
except ImportError:  # direct module import (src/ on sys.path)
    from adapter import adapt_building, adapt_land_use, load_hadian_submission
    from pipeline import (TechnicalReviewRunner, check_living_circle,
                          check_setback, check_solar, check_sponge_city,
                          check_view_corridor)

from shapely.geometry import shape as shapely_shape

DEFAULT_SITE_AREA_SQM = 11_400_000.0  # haidian declared site area fallback


def _site_area_sqm(site_boundary: Optional[Dict[str, Any]]) -> float:
    """Site area in sqm: declared value, else projected boundary area."""
    if not site_boundary:
        return DEFAULT_SITE_AREA_SQM
    props = site_boundary.get('properties') or {}
    declared = (props.get('area_sqm_declared')
                or props.get('area_sqm_calculated'))
    if isinstance(declared, (int, float)) and declared > 0:
        return float(declared)
    geom = site_boundary.get('geometry')
    if geom is not None:
        try:
            from .adapter import _projected
            g = _projected(shapely_shape(geom))
            if g is not None and g.area > 0:
                return float(g.area)
        except Exception:
            pass
    return DEFAULT_SITE_AREA_SQM


def run_technical_review(submission_path: str,
                         out_path: Optional[str] = None) -> Dict[str, Any]:
    """Run all technical checks on a haidian submission end-to-end.

    1. Load the submission's geometry layers via :func:`adapter.load_hadian_submission`.
    2. Adapt buildings and land-use features into pipeline check inputs
       (heights, solar spacing and setbacks measured from geometry).
    3. Run the registered technical checks through
       :class:`TechnicalReviewRunner` — solar access, view corridor
       (NOT_ASSESSED without official corridor data), sponge city
       volume, building setback, 15-minute living circle.
    4. Save the aggregated report (gate decision + per-check evidence
       trail + adapter meta) as ``tech_review.json`` — by default in the
       submission directory itself.
    5. Return the report.

    Parameters
    ----------
    submission_path : str
        Submission root (containing ``geometry/``) or the geometry
        directory itself.
    out_path : str, optional
        Override the report output path (default:
        ``<submission_path>/tech_review.json``).

    Returns
    -------
    dict
        The report written to disk: keys ``gate_passed``, ``total_checks``,
        ``passed``, ``failed``, ``not_assessed``, ``blocking_failures``,
        ``not_assessed_checks``, ``results`` and ``meta``.
    """
    layers = load_hadian_submission(submission_path)

    raw_buildings = layers.get('buildings', [])
    buildings = [
        adapt_building(f,
                       neighbors=raw_buildings,
                       roads=layers.get('roads'),
                       site_boundary=layers.get('site_boundary'))
        for f in raw_buildings
    ]
    land_use = [adapt_land_use(f) for f in layers.get('land_use', [])]
    site_area_sqm = _site_area_sqm(layers.get('site_boundary'))

    runner = TechnicalReviewRunner()
    for check_fn in (check_solar, check_view_corridor, check_sponge_city,
                     check_setback, check_living_circle):
        runner.register(check_fn)

    report = runner.run_all(
        buildings=buildings,
        land_use_features=land_use,
        corridors_defined=False,        # official corridor GIS not shipped
        facility_counts={},             # no facility inventory in geometry/
        residential_pop=0,              # no population data in geometry/
        site_area_sqm=site_area_sqm,
    )

    report['meta'] = {
        'tool': 'urban-spatial-tooling',
        'module': 'src/integration.py',
        'submission_path': str(Path(submission_path).resolve()),
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'layer_counts': {
            key: (len(value) if isinstance(value, list)
                  else (1 if value is not None else 0))
            for key, value in layers.items()
        },
        'gate_semantics': ('only CRITICAL/MAJOR failures block the gate; '
                           'NOT_ASSESSED checks never block but are reported'),
    }

    out = Path(out_path) if out_path else Path(submission_path) / 'tech_review.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


__all__ = ["run_technical_review"]
