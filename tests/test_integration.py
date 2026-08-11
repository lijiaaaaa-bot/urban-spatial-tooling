"""End-to-end tests for src/integration.py — technical review of a
synthetic haidian submission."""

import json
from pathlib import Path

import pytest
from pyproj import Transformer

from src.integration import run_technical_review

# ---------------------------------------------------------------------------
# Helpers: meter-exact WGS84 geometry around (116.35, 40.0)
#
# Geometry is authored in EPSG:4548 metre offsets and inverse-projected
# to WGS84, so the adapter's forward projection returns the exact
# authored distances.
# ---------------------------------------------------------------------------

LON0, LAT0 = 116.35, 40.0
_LONLAT_TO_M = Transformer.from_crs('EPSG:4326', 'EPSG:4548', always_xy=True)
_M_TO_LONLAT = Transformer.from_crs('EPSG:4548', 'EPSG:4326', always_xy=True)
_CX, _CY = _LONLAT_TO_M.transform(LON0, LAT0)


def m_to_deg(x_m: float, y_m: float):
    return _M_TO_LONLAT.transform(_CX + x_m, _CY + y_m)


def box_feature(fid: str, x0: float, y0: float, x1: float, y1: float,
                props: dict) -> dict:
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    coords = [[m_to_deg(x, y) for x, y in corners]]
    return {
        'type': 'Feature', 'id': fid,
        'properties': {'id': fid, **props},
        'geometry': {'type': 'Polygon', 'coordinates': coords},
    }


def line_feature(fid: str, x0: float, y0: float, x1: float, y1: float,
                 props: dict) -> dict:
    return {
        'type': 'Feature', 'id': fid,
        'properties': {'id': fid, **props},
        'geometry': {'type': 'LineString',
                     'coordinates': [m_to_deg(x0, y0), m_to_deg(x1, y1)]},
    }


def write_collection(path: Path, features: list) -> None:
    path.write_text(json.dumps({'type': 'FeatureCollection',
                                'features': features}), encoding='utf-8')


def build_submission(root: Path, *, compliant: bool = True) -> Path:
    """Create a synthetic haidian submission under ``root/submission``.

    Layout (metre offsets): a 600 m x 600 m site with green land use on
    the north half and residential on the south half; three 40 m x 40 m
    buildings in a column (residential, residential, office); one E-W
    arterial road south of everything.

    Compliant: 50 m solar gap between the two residential buildings.
    Non-compliant: the middle building slides north until its N-S
    extent overlaps the north building (solar spacing 0 -> CRITICAL).
    """
    sub = root / 'submission'
    geo = sub / 'geometry'
    geo.mkdir(parents=True)

    site = box_feature('SITE-001', -300, -300, 300, 300,
                       {'layer': 'SITE_BOUNDARY',
                        'area_sqm_declared': 360000.0})
    write_collection(geo / 'site_boundary.geojson', [site])

    green = box_feature('LU-001', -300, 0, 300, 300,
                        {'layer': 'LAND_USE', 'land_use_code': '1401'})
    residential = box_feature('LU-002', -300, -300, 300, 0,
                              {'layer': 'LAND_USE', 'land_use_code': '0701'})
    write_collection(geo / 'land_use.geojson', [green, residential])

    b1 = box_feature('BLDG-001', -20, 130, 20, 170,
                     {'layer': 'BUILDING_FOOTPRINT',
                      'building_type': 'residential', 'floors_estimated': 6})
    b2_y0 = 105 if not compliant else 40
    b2 = box_feature('BLDG-002', -20, b2_y0, 20, b2_y0 + 40,
                     {'layer': 'BUILDING_FOOTPRINT',
                      'building_type': 'residential', 'floors_estimated': 6})
    b3 = box_feature('BLDG-003', -20, -140, 20, -100,
                     {'layer': 'BUILDING_FOOTPRINT',
                      'building_type': 'office', 'floors_estimated': 6})
    write_collection(geo / 'buildings.geojson', [b1, b2, b3])

    road = line_feature('ROAD-001', -300, -200, 300, -200,
                        {'layer': 'ROAD_CENTERLINE', 'road_type': 'arterial'})
    write_collection(geo / 'roads.geojson', [road])

    return sub


def find_result(report: dict, check_id: str) -> dict:
    return next(r for r in report['results'] if r['check_id'] == check_id)


# ---------------------------------------------------------------------------
# Compliant scenario: full pipeline, gate PASS
# ---------------------------------------------------------------------------


def test_run_technical_review_compliant_scenario_passes(tmp_path):
    """A compliant synthetic submission passes the gate end-to-end and
    the report is persisted as tech_review.json in the submission dir."""
    sub = build_submission(tmp_path, compliant=True)
    report = run_technical_review(str(sub))

    assert report['gate_passed'] is True
    assert report['blocking_failures'] == 0

    # solar: 50 m gap >= 1.6 * 18 m = 28.8 m
    solar = find_result(report, 'CHECK-SOLAR-001')
    assert solar['passed'] is True
    # setback: buildings >= 60 m from the arterial road (15 m required)
    setback = find_result(report, 'CHECK-SETBACK-001')
    assert setback['passed'] is True
    # sponge: 180,000 m2 green -> LID volume covers the 85% control rate
    sponge = find_result(report, 'CHECK-SPONGE-001')
    assert sponge['passed'] is True
    # view corridor: data-gated, honestly NOT_ASSESSED, never blocks
    view = find_result(report, 'CHECK-VIEW-001')
    assert view['severity'] == 'not_assessed'
    assert view['passed'] is False
    assert report['not_assessed'] == 1

    # report persisted at <submission_path>/tech_review.json
    out = sub / 'tech_review.json'
    assert out.exists()
    with open(out, encoding='utf-8') as f:
        written = json.load(f)
    assert written == report
    assert written['meta']['layer_counts'] == {
        'buildings': 3, 'land_use': 2, 'site_boundary': 1, 'roads': 1,
    }


def test_run_technical_review_violating_scenario_fails_gate(tmp_path):
    """Overlapping N-S extents -> solar spacing 0 -> CRITICAL failure
    blocks the gate; the honest report still gets written."""
    sub = build_submission(tmp_path, compliant=False)
    report = run_technical_review(str(sub))

    assert report['gate_passed'] is False
    assert report['blocking_failures'] == 1

    solar = find_result(report, 'CHECK-SOLAR-001')
    assert solar['passed'] is False
    assert solar['severity'] == 'critical'
    assert solar['evidence']['violations'] >= 1

    assert (sub / 'tech_review.json').exists()


def test_run_technical_review_accepts_geometry_dir_and_out_path(tmp_path):
    """The geometry directory itself works as submission_path, and
    out_path overrides where the report lands."""
    sub = build_submission(tmp_path, compliant=True)
    out = tmp_path / 'reports' / 'tech_review.json'
    report = run_technical_review(str(sub / 'geometry'), out_path=str(out))
    assert report['gate_passed'] is True
    assert out.exists()
    assert not (sub / 'tech_review.json').exists()
