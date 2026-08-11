"""Tests for src/solar.py — sun position and daylight spacing.

Verified against known physical values for Beijing (lat 39.9 N):

- winter solstice (Dec 21) solar noon altitude   ~ 26.7 deg
- 大寒日 (Jan 20, statutory day, GB 50180-2018)    ~ 29.8 deg
- summer solstice (Jun 21) solar noon altitude    ~ 73.6 deg
- winter solstice 09:00 azimuth                   ~ 138 deg (southeast)
"""

import math

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from src.solar import (
    BEIJING_LAT,
    BEIJING_LON,
    BEIJING_TZ,
    D_H_STANDARD_LOW,
    GREAT_COLD,
    SUMMER_SOLSTICE,
    WINTER_SOLSTICE,
    H_SILL,
    _facade_edges,
    _sample_window_positions,
    building_shadow_at_height,
    check_per_window_compliance,
    compute_spacing_d,
    facade_insolation,
    sun_position,
    window_receives_sun,
)


@pytest.fixture
def winter_noon():
    """Solar position at Beijing winter-solstice noon."""
    return sun_position(BEIJING_LAT, BEIJING_LON, BEIJING_TZ, WINTER_SOLSTICE, 12.0)


def test_winter_solstice_noon_altitude(winter_noon):
    """h = 90 - 39.9 - 23.44 = 26.66 deg."""
    assert winter_noon.altitude == pytest.approx(26.7, abs=0.1)


def test_summer_solstice_noon_altitude():
    pos = sun_position(BEIJING_LAT, BEIJING_LON, BEIJING_TZ, SUMMER_SOLSTICE, 12.0)
    assert pos.altitude == pytest.approx(73.6, abs=0.1)


def test_great_cold_noon_altitude():
    """大寒日 (Jan 20) — the statutory analysis day (GB 50180-2018 表 4.0.9)."""
    pos = sun_position(BEIJING_LAT, BEIJING_LON, BEIJING_TZ, GREAT_COLD, 12.0)
    assert pos.altitude == pytest.approx(29.8, abs=0.05)


def test_morning_azimuth_is_southeast():
    """09:00 on the winter solstice: sun in the SE (~138 deg), i.e. strictly
    between E (90) and S (180) — NOT NE and NOT NW."""
    pos = sun_position(BEIJING_LAT, BEIJING_LON, BEIJING_TZ, WINTER_SOLSTICE, 9.0)
    assert 90.0 < pos.azimuth < 180.0
    assert pos.azimuth == pytest.approx(138.0, abs=2.0)


def test_afternoon_azimuth_is_southwest():
    """15:00: sun in the SW (~222 deg); azimuth is clockwise from north."""
    pos = sun_position(BEIJING_LAT, BEIJING_LON, BEIJING_TZ, WINTER_SOLSTICE, 15.0)
    assert 180.0 < pos.azimuth < 270.0
    assert pos.azimuth == pytest.approx(222.0, abs=2.0)


def test_shadow_length_ratio(winter_noon):
    """Shadow/H = 1/tan(alt); winter noon ~1.99, summer noon ~0.29."""
    summer_noon = sun_position(BEIJING_LAT, BEIJING_LON, BEIJING_TZ, SUMMER_SOLSTICE, 12.0)
    assert winter_noon.shadow_length_ratio == pytest.approx(1.99, rel=0.02)
    assert summer_noon.shadow_length_ratio == pytest.approx(0.29, rel=0.02)


def test_compute_spacing_d_known_value():
    """H = 18 m, winter solstice 09:00: D ~ 50.8 m -> D/H ~ 2.82, an upper
    bound above the Beijing 1.6-1.7 practice standard (大寒日-calibrated)."""
    D = compute_spacing_d(18.0, WINTER_SOLSTICE, 9.0)
    assert D == pytest.approx(50.8, abs=0.5)
    assert D / 18.0 > D_H_STANDARD_LOW


def test_compute_spacing_d_infinite_at_night():
    """Sun below the horizon: no spacing analysis possible."""
    assert math.isinf(compute_spacing_d(18.0, WINTER_SOLSTICE, 0.0))


# ---------------------------------------------------------------------------
# Per-window facade insolation (窗台面日照, GB 50180-2018 表 4.0.9)
# ---------------------------------------------------------------------------


def _rect(x0, y0, x1, y1):
    """Rectangular building outline (x east, y north), CCW."""
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def test_building_shadow_at_height_sweep_semantics():
    """The shadow region at height z is the footprint swept along the shadow
    direction: at 10:00 on 大寒日 (sun in the SE) a point NW of the building
    is in shadow while a point on the sun side is not; the footprint itself
    is always covered.  Windows at/above the roof are never blocked."""
    outline = _rect(0, 0, 30, 12)
    pos = sun_position(BEIJING_LAT, BEIJING_LON, BEIJING_TZ, GREAT_COLD, 10.0)
    shadow = building_shadow_at_height(outline, 18.0, pos, 0.0)
    assert shadow is not None
    poly = Polygon(shadow)
    assert poly.contains(Point(5, 30))        # NW: inside the shadow sweep
    assert not poly.contains(Point(35, 10))   # SE: sun side, clear
    assert poly.covers(Polygon(outline))      # footprint always covered
    # at/above the roof there is nothing to block
    assert building_shadow_at_height(outline, 18.0, pos, 18.0) is None
    assert building_shadow_at_height(outline, 18.0, pos, 30.0) is None


def test_facade_insolation_unobstructed_building():
    """Stand-alone building on 大寒日 8:00-16:00 (half-open window): the
    south facade keeps the full 8 h, north gets 0 h (sun never in the north
    half-space in winter), east/west catch morning/afternoon sun — and the
    building's own shadow never blocks its own facade."""
    res = facade_insolation([(_rect(0, 0, 30, 12), 18.0)],
                            day=GREAT_COLD, hours_range=(8.0, 16.0),
                            time_step_min=30)
    facades = {f['orientation']: f for f in res[0]['facades']}
    assert set(facades) == {0.0, 90.0, 180.0, 270.0}
    assert facades[180.0]['min_hours'] == pytest.approx(8.0)   # south: full window
    assert facades[0.0]['min_hours'] == 0.0                    # north: geometric 0
    assert facades[90.0]['min_hours'] == pytest.approx(4.0, abs=0.01)
    assert facades[270.0]['min_hours'] == pytest.approx(3.5, abs=0.01)
    # every sampled window sits at or above the 0.9 m sill (GB 50180-2018)
    assert all(f['n_windows'] == len(f['windows'])
               and min(f['windows']) >= 0.0 for f in res[0]['facades'])
    assert all(w[2] >= H_SILL
               for f in res[0]['facades']
               for w in _sample_window_positions(
                   *_facade_edges(_rect(0, 0, 30, 12))[0], 18.0,
                   H_SILL, 1.5, 1.2))


def test_facade_insolation_neighbor_blocking():
    """A tall south slab (H=30 m) at 28 m spacing shades the north slab's
    south-facing windows: the critical (worst) window falls below 2 h on
    大寒日 while the tall building's own south facade keeps the full 8 h —
    a building's own shadow never counts.  The independent ray-cast agrees
    with the vectorized shadow sweep on the critical window."""
    south = _rect(10, 90, 80, 102)
    north = _rect(10, 130, 80, 142)
    buildings = [(south, 30.0), (north, 18.0)]
    res = facade_insolation(buildings, day=GREAT_COLD, hours_range=(8.0, 16.0),
                            time_step_min=30)
    s_facades = {f['orientation']: f for f in res[0]['facades']}
    n_facades = {f['orientation']: f for f in res[1]['facades']}

    # own shadow does not count: the south building keeps 8 h on its south face
    assert s_facades[180.0]['min_hours'] == pytest.approx(8.0)
    # north building's south facade: critical window below the 2 h standard
    assert n_facades[180.0]['min_hours'] < 2.0
    # ...but its east/west ends still catch morning/afternoon sun
    assert n_facades[90.0]['min_hours'] > 2.0
    assert n_facades[270.0]['min_hours'] > 2.0

    # compliance check flags the north building's south facade
    check = check_per_window_compliance(res, min_hours=2.0)
    assert not check.passes
    south_fail = next(d for d in check.deficient_facades if d['orientation'] == 180.0)
    assert south_fail['building_id'] == 1

    # ray-cast reference agrees with the vectorized sweep on the critical window
    f = n_facades[180.0]
    wi = f['min_window_index']
    p0, p1 = _facade_edges(north)[0]
    wx, wy, wz = _sample_window_positions(p0, p1, 18.0, H_SILL, 1.5, 1.2)[wi]
    hours_rc = 0.0
    for k in range(16):
        hour = 8.0 + k * 0.5
        pos = sun_position(BEIJING_LAT, BEIJING_LON, BEIJING_TZ, GREAT_COLD, hour)
        if window_receives_sun(wx, wy, wz, pos, 180.0, north, buildings):
            hours_rc += 0.5
    assert hours_rc == pytest.approx(f['windows'][wi], abs=1e-9)
    assert hours_rc < 2.0


def test_check_per_window_compliance():
    """Hand-built results: pass/fail accounting, per-facade detail, summary."""
    ok = {
        0: {'min_hours': 5.0, 'n_windows': 2,
            'facades': [{'orientation': 180.0, 'min_hours': 5.0,
                         'windows': [5.0, 6.0]}]},
    }
    good = check_per_window_compliance(ok)
    assert good.passes
    assert good.total_windows == 2
    assert good.deficient_windows == 0
    assert good.deficient_facades == []
    assert 'PASS' in good.summary

    bad = {
        0: {'min_hours': 0.5, 'n_windows': 4,
            'facades': [{'orientation': 180.0, 'min_hours': 0.5,
                         'windows': [0.5, 1.0, 2.5, 3.0]}]},
        1: {'min_hours': 0.0, 'n_windows': 2,
            'facades': [{'orientation': 90.0, 'min_hours': 0.0,
                         'windows': [0.0, 1.9]}]},
    }
    check = check_per_window_compliance(bad, min_hours=2.0)
    assert not check.passes
    assert check.total_windows == 6
    assert check.deficient_windows == 4            # 0.5, 1.0, 0.0, 1.9
    assert len(check.deficient_facades) == 2
    assert [f['building_id'] for f in check.deficient_facades] == [0, 1]
    assert check.deficient_facades[1]['orientation'] == 90.0
    assert 'FAIL' in check.summary
