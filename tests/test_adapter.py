"""Tests for src/adapter.py — haidian feature -> pipeline check inputs."""

import json
from pathlib import Path

import pytest
from pyproj import Transformer

from src.adapter import adapt_building, adapt_land_use, load_hadian_submission

# ---------------------------------------------------------------------------
# Helpers: build meter-exact WGS84 rectangles around (116.35, 40.0)
#
# Geometry is authored in EPSG:4548 metre offsets and inverse-projected
# to WGS84, so the adapter's forward projection returns the exact
# authored distances (no ellipsoid-constant approximation).
# ---------------------------------------------------------------------------

from pyproj import Transformer

LON0, LAT0 = 116.35, 40.0
_LONLAT_TO_M = Transformer.from_crs('EPSG:4326', 'EPSG:4548', always_xy=True)
_M_TO_LONLAT = Transformer.from_crs('EPSG:4548', 'EPSG:4326', always_xy=True)
_CX, _CY = _LONLAT_TO_M.transform(LON0, LAT0)


def m_to_deg(x_m: float, y_m: float):
    """Metre offsets from (LON0, LAT0) -> (lon, lat) degrees."""
    return _M_TO_LONLAT.transform(_CX + x_m, _CY + y_m)


def box_feature(fid: str, x0: float, y0: float, x1: float, y1: float,
                props: dict) -> dict:
    """GeoJSON rectangle feature with 40 m-wide footprint, WGS84 geometry."""
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    coords = [[m_to_deg(x, y) for x, y in corners]]
    return {
        'type': 'Feature', 'id': fid,
        'properties': {'id': fid, **props},
        'geometry': {'type': 'Polygon', 'coordinates': coords},
    }


def line_feature(fid: str, x0: float, y0: float, x1: float, y1: float,
                 props: dict) -> dict:
    """GeoJSON LineString feature in metre-offset space."""
    return {
        'type': 'Feature', 'id': fid,
        'properties': {'id': fid, **props},
        'geometry': {'type': 'LineString',
                     'coordinates': [m_to_deg(x0, y0), m_to_deg(x1, y1)]},
    }


# Realistic haidian test fixture: a building 40 m deep with a south
# neighbor 80 m away, an arterial road 180 m to the south, and a site
# boundary around everything.
SITE = box_feature('SITE-001', -300, -300, 300, 300,
                   {'layer': 'SITE_BOUNDARY', 'area_sqm_declared': 360000.0})
NORTH_BLDG = box_feature('BLDG-001', -20, 100, 20, 140,
                         {'layer': 'BUILDING_FOOTPRINT',
                          'building_type': 'residential',
                          'floors_estimated': 6})
SOUTH_BLDG = box_feature('BLDG-002', -20, -20, 20, 20,
                         {'layer': 'BUILDING_FOOTPRINT',
                          'building_type': 'residential',
                          'floors_estimated': 6})
ROAD = line_feature('ROAD-001', -300, -80, 300, -80,
                    {'layer': 'ROAD_CENTERLINE', 'road_type': 'major'})


# ---------------------------------------------------------------------------
# adapt_building
# ---------------------------------------------------------------------------


def test_adapt_building_estimates_height_and_measures_spacing_setback():
    """A realistic haidian building (no computed attrs) gets all four
    pipeline attributes: height from storeys, spacing and setback from
    geometry."""
    adapted = adapt_building(NORTH_BLDG,
                             neighbors=[SOUTH_BLDG],
                             roads=[ROAD],
                             site_boundary=SITE)
    props = adapted['properties']

    # height: 6 storeys * 3.0 m/floor (residential)
    assert props['height_m'] == 18.0
    assert props['height_source'] == 'estimated_from_storeys'
    # solar spacing: 80 m clear N-S gap to the south neighbor
    assert props['spacing_to_south_m'] == pytest.approx(80.0, abs=0.5)
    assert props['spacing_source'] == 'geometry_nearest_south'
    # setback: 180 m to the road, 'major' normalized to pipeline 'arterial'
    assert props['min_setback_m'] == pytest.approx(180.0, abs=0.5)
    assert props['adjacent_road_class'] == 'arterial'
    # identity passthrough
    assert props['building_type'] == 'residential'
    assert props['building_type_original'] == 'residential'
    assert props['id'] == 'BLDG-001'
    assert props['floors_estimated'] == 6
    # footprint area computed in projected metres (40 m x 40 m)
    assert props['area_sqm'] == pytest.approx(1600.0, rel=0.01)
    assert props['_adapter_notes']


def test_adapt_building_without_context_leaves_values_absent():
    """No neighbors / roads / boundary -> spacing stays absent (honest,
    the pipeline fails closed on missing solar data) and no setback
    attribute is fabricated."""
    adapted = adapt_building(NORTH_BLDG)
    props = adapted['properties']
    assert props['spacing_to_south_m'] is None
    assert props['spacing_source'] is None
    assert 'min_setback_m' not in props
    assert 'adjacent_road_class' not in props
    assert props['height_m'] == 18.0  # storeys still estimable


def test_adapt_building_site_boundary_fallback_for_southmost_building():
    """A building with no south neighbor is spaced against the site's
    south edge — no in-submission obstruction, so the constraint is
    bounded only by the site edge."""
    lone = box_feature('BLDG-003', -20, 150, 20, 190,
                       {'layer': 'BUILDING_FOOTPRINT',
                        'building_type': 'residential',
                        'floors_estimated': 6})
    adapted = adapt_building(lone, neighbors=[], site_boundary=SITE)
    props = adapted['properties']
    # south edge (150 m) to site south edge (-300 m) = 450 m
    assert props['spacing_to_south_m'] == pytest.approx(450.0, abs=0.5)
    assert props['spacing_source'] == 'geometry_site_boundary_fallback'


def test_adapt_building_talent_apartment_counts_as_residential():
    """talent_apartment is housing: normalized to 'residential' so the
    solar check applies to it."""
    apt = box_feature('BLDG-004', -20, 10, 20, 50,
                      {'layer': 'BUILDING_FOOTPRINT',
                       'building_type': 'talent_apartment',
                       'floors_estimated': 6})
    adapted = adapt_building(apt)
    assert adapted['properties']['building_type'] == 'residential'
    assert adapted['properties']['building_type_original'] == 'talent_apartment'


# ---------------------------------------------------------------------------
# adapt_land_use
# ---------------------------------------------------------------------------


def test_adapt_land_use_maps_code_and_uses_declared_area():
    """0701 (城镇住宅用地) -> residential; declared area is preserved."""
    feature = box_feature('LU-001', 0, 0, 100, 100,
                          {'layer': 'LAND_USE', 'land_use_code': '0701',
                           'area_sqm_declared': 9876.5})
    adapted = adapt_land_use(feature)
    props = adapted['properties']
    assert props['land_use_type'] == 'residential'
    assert props['land_use_code'] == '0701'
    assert props['area_sqm'] == 9876.5
    assert props['area_source'] == 'declared'


def test_adapt_land_use_computes_area_from_geometry():
    """1401 (公园绿地) -> green; without a declared area the EPSG:4548
    projected area is computed (100 m x 100 m box)."""
    feature = box_feature('LU-002', 0, 0, 100, 100,
                          {'layer': 'LAND_USE', 'land_use_code': '1401'})
    adapted = adapt_land_use(feature)
    props = adapted['properties']
    assert props['land_use_type'] == 'green'
    assert props['area_sqm'] == pytest.approx(10_000.0, rel=0.01)
    assert props['area_source'] == 'geometry_projected'


def test_adapt_land_use_unknown_code_falls_back_to_default():
    """Unmapped codes fall back to the pipeline default phi (residential)."""
    feature = box_feature('LU-003', 0, 0, 100, 100,
                          {'layer': 'LAND_USE', 'land_use_code': '9999'})
    adapted = adapt_land_use(feature)
    assert adapted['properties']['land_use_type'] == 'residential'
    assert adapted['properties']['_adapter_notes']


# ---------------------------------------------------------------------------
# load_hadian_submission
# ---------------------------------------------------------------------------


def write_collection(path: Path, features: list) -> None:
    collection = {'type': 'FeatureCollection', 'features': features}
    path.write_text(json.dumps(collection), encoding='utf-8')


def test_load_hadian_submission_synthetic_fixture(tmp_path):
    """Loader buckets geometry/*.geojson into canonical layer keys and
    picks a single site boundary feature."""
    geo = tmp_path / 'submission' / 'geometry'
    geo.mkdir(parents=True)
    write_collection(geo / 'site_boundary.geojson', [SITE])
    write_collection(geo / 'land_use.geojson',
                     [box_feature('LU-A', 0, 0, 10, 10,
                                  {'layer': 'LAND_USE', 'land_use_code': '0701'})])
    write_collection(geo / 'buildings.geojson', [NORTH_BLDG])
    write_collection(geo / 'roads.geojson', [ROAD])
    write_collection(geo / 'green_space.geojson',
                     [box_feature('GS-A', 0, 0, 10, 10,
                                  {'layer': 'GREEN_SPACE'})])

    layers = load_hadian_submission(str(tmp_path / 'submission'))
    assert len(layers['buildings']) == 1
    assert len(layers['land_use']) == 1
    assert len(layers['roads']) == 1
    assert layers['site_boundary']['id'] == 'SITE-001'
    assert len(layers['green_space']) == 1
    assert layers['site_boundary']['properties']['layer'] == 'SITE_BOUNDARY'

    # the geometry directory itself is also accepted
    direct = load_hadian_submission(str(geo))
    assert len(direct['buildings']) == 1


def test_load_hadian_submission_real_submission():
    """The haidian test submission loads with the expected layer sizes."""
    real = Path('/Users/lijia/Projects/haidian/submissions/test/test')
    if not (real / 'geometry').is_dir():
        pytest.skip('haidian repo test submission not present')
    layers = load_hadian_submission(str(real))
    assert len(layers['buildings']) == 12
    assert len(layers['land_use']) == 25
    assert len(layers['roads']) == 13
    assert layers['site_boundary'] is not None
    assert layers['site_boundary']['id'] == 'SITE-001'
    assert len(layers['green_space']) > 0
    # every building carries haidian's attribute vocabulary
    for f in layers['buildings']:
        props = f['properties']
        assert props['layer'] == 'BUILDING_FOOTPRINT'
        assert 'building_type' in props
        assert 'height_m' not in props          # pipeline attrs absent
        assert 'spacing_to_south_m' not in props
