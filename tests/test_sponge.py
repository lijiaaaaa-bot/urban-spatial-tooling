"""Tests for src/sponge.py — 海绵城市 容积法 storage + continuous simulation."""

import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import box

from src.sponge import (
    design_rainfall_mm,
    required_storage_m3,
    sponge_city_check,
    chicago_hyetograph,
    water_balance_step,
    continuous_simulation,
)


def test_design_rainfall_mm_085_is_33_6():
    """DB11/685-2021 Beijing new-development design point: 85% -> 33.6 mm."""
    assert design_rainfall_mm(0.85) == 33.6


def test_required_storage_m3_known_inputs():
    """V = 10 x H x phi x F: 33.6 mm x 0.6 x 10 ha = 2016 m3."""
    assert required_storage_m3(33.6, 0.6, 10.0) == pytest.approx(2016.0)
    # 21.4 mm (75% control rate) x 0.53 x 2 ha = 226.84 m3
    assert required_storage_m3(21.4, 0.53, 2.0) == pytest.approx(226.84)


def test_higher_control_rate_requires_more_storage():
    """Volume scales linearly with the design rainfall depth lookup."""
    parcels = gpd.GeoDataFrame(
        [
            {"land_use_code": "R", "area_sqm": 10000.0, "geometry": box(0, 0, 100, 100)},
            {"land_use_code": "G", "area_sqm": 5000.0, "geometry": box(100, 0, 200, 100)},
        ],
        crs="EPSG:4548",
    )
    v_high = sponge_city_check(parcels, control_rate=1.0)  # clamps to 85% -> 33.6 mm
    v_low = sponge_city_check(parcels, control_rate=0.75)  # 21.4 mm
    assert v_high["design_rainfall_mm"] == 33.6
    assert v_low["design_rainfall_mm"] == 21.4
    assert v_high["required_storage_m3"] > v_low["required_storage_m3"]
    assert v_high["required_storage_m3"] == pytest.approx(
        v_low["required_storage_m3"] * 33.6 / 21.4
    )


# ── Continuous simulation (Chicago hyetograph + water balance) ──────


def test_chicago_hyetograph_total_depth():
    """24 h / 5 min storm rescales to exactly the design depth."""
    h = chicago_hyetograph(duration_min=1440, dt_min=5, P_total_mm=45.0)
    assert h["depth_mm"].sum() == pytest.approx(45.0, abs=1e-9)
    assert h["P_total_mm"] == pytest.approx(45.0)
    assert len(h["depth_mm"]) == 288  # 1440 / 5
    assert np.all(h["depth_mm"] >= 0.0)


def test_chicago_hyetograph_peak_at_r_times_T():
    """Beijing peak ratio r=0.4: peak step sits at t_p = 0.4 x 1440 = 576 min."""
    h = chicago_hyetograph(duration_min=1440, dt_min=5, P_total_mm=45.0,
                           r=0.4)
    peak_i = int(np.argmax(h["depth_mm"]))
    assert abs(h["times_min"][peak_i] - h["t_peak_min"]) <= h["dt_min"]
    # rising limb non-decreasing, falling limb non-increasing
    depths = h["depth_mm"]
    assert np.all(np.diff(depths[:peak_i + 1]) >= -1e-9)
    assert np.all(np.diff(depths[peak_i:]) <= 1e-9)


def test_chicago_hyetograph_scaling_shape_invariant():
    """Doubling the design depth scales the hyetograph, not its shape."""
    h1 = chicago_hyetograph(duration_min=1440, dt_min=5, P_total_mm=45.0)
    h2 = chicago_hyetograph(duration_min=1440, dt_min=5, P_total_mm=90.0)
    np.testing.assert_allclose(h2["depth_mm"] / 90.0, h1["depth_mm"] / 45.0,
                               rtol=1e-12)


def test_water_balance_step_conservation():
    """rainfall = infiltration + runoff + dS within one step."""
    S0 = 10.0
    S1, r = water_balance_step(S0, 3.0, dt_min=5, phi=0.53,
                               f_inf_mm=1.5, S_cap_mm=20.0,
                               q_release_mm_h=6.0)
    dS = r["storage_after_mm"] - S0
    assert r["rainfall_mm"] == pytest.approx(
        r["infiltration_mm"] + r["runoff_mm"] + dS, abs=1e-9)
    assert S1 == r["storage_after_mm"]


def test_water_balance_step_dry_period_drains_storage():
    """No rain: storage loses release + infiltration; runoff = release only."""
    S1, r = water_balance_step(10.0, 0.0, dt_min=5, phi=0.53,
                               f_inf_mm=2.0, S_cap_mm=50.0,
                               q_release_mm_h=6.0)
    # release = 6 mm/h x 5/60 h = 0.5 mm; infiltration = 2 mm
    assert r["storage_after_mm"] == pytest.approx(7.5)
    assert r["runoff_mm"] == pytest.approx(0.5)
    assert r["overflow_mm"] == 0.0


def test_water_balance_step_capacity_caps_storage():
    """Storage is capped at S_cap; the excess overflows."""
    S1, r = water_balance_step(0.0, 5.0, dt_min=5, phi=0.53,
                               f_inf_mm=0.0, S_cap_mm=1.0,
                               q_release_mm_h=0.0)
    assert r["storage_after_mm"] == pytest.approx(1.0)
    assert r["overflow_mm"] == pytest.approx(5.0 - 1.0)
    assert r["runoff_mm"] == pytest.approx(4.0)


def test_continuous_simulation_reduces_peak():
    """LID storage cuts the peak outflow vs the no-LID baseline."""
    h = chicago_hyetograph(duration_min=1440, dt_min=5, P_total_mm=85.0)
    no_lid = continuous_simulation(h["depth_mm"], dt_min=5, F_ha=100.0,
                                   phi=0.6, S_cap_m3=0.0,
                                   q_release_mm_h=0.0)
    lid = continuous_simulation(h["depth_mm"], dt_min=5, F_ha=100.0,
                                phi=0.6, S_cap_m3=5000.0,
                                q_release_mm_h=5.0)
    assert lid["summary"]["peak_runoff_m3_s"] < (
        no_lid["summary"]["peak_runoff_m3_s"])


def test_continuous_simulation_mass_closure():
    """Total rainfall = infiltration + runoff + final storage (closed)."""
    h = chicago_hyetograph(duration_min=720, dt_min=5, P_total_mm=33.6)
    sim = continuous_simulation(h["depth_mm"], dt_min=5, F_ha=1141.28,
                                phi=0.53, S_cap_m3=202473.0)
    rain = sim["summary"]["total_rainfall_mm"]
    inf = sim["infiltration_mm"].sum()
    run = sim["runoff_mm"].sum()
    dS = sim["storage_mm"][-1] - 0.0
    assert rain == pytest.approx(inf + run + dS, abs=1e-6)
    assert sim["summary"]["capture_rate"] == pytest.approx(
        1.0 - sim["summary"]["runoff_coefficient"])
