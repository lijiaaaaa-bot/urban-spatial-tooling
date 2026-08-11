"""Tests for src/vertical.py — vertical design / grading (竖向设计)."""

import numpy as np
import pytest

from src.vertical import (
    D8_DELTA,
    cut_fill_balance,
    flow_direction_d8,
    slope_aspect,
)


def test_slope_aspect_ramp_orientation():
    # Grid convention: row 0 = north, col 0 = west, dx = dy = 1.
    i, j = np.meshgrid(np.arange(5.0), np.arange(5.0))

    # ramp rising east AND south: gradient 0.5 -> 50 % slope / 26.565 deg
    z = 0.3 * i + 0.4 * j
    r = slope_aspect(z)
    assert r["slope_pct"][2, 2] == pytest.approx(50.0)
    assert r["slope_deg"][2, 2] == pytest.approx(np.degrees(np.arctan(0.5)))
    # uphill = SE -> downhill azimuth 323.13 deg = NW
    assert r["aspect_deg"][2, 2] == pytest.approx(323.13010235415598, abs=1e-9)
    assert r["aspect_dir"][2, 2] == "NW"

    # pure eastward rise -> downhill west; pure southward rise -> downhill north
    r2 = slope_aspect(0.3 * i)
    assert r2["aspect_deg"][2, 2] == pytest.approx(270.0)
    assert r2["aspect_dir"][2, 2] == "W"
    r3 = slope_aspect(0.4 * j)
    assert r3["aspect_deg"][2, 2] == pytest.approx(0.0)
    assert r3["aspect_dir"][2, 2] == "N"

    # constant DEM -> flat everywhere, aspect undefined
    rf = slope_aspect(np.zeros((4, 4)))
    assert rf["slope_pct"].max() == 0.0
    assert rf["aspect_dir"][0, 0] == "flat"
    assert rf["aspect_deg"][0, 0] == -1.0


def test_cut_fill_balance_volumes_and_optimum():
    # 2 low + 2 high cells, 10 m relief, 1 m cells
    z = np.array([[0.0, 0.0], [10.0, 10.0]])

    # platform at 5 m: two cells cut 5 m each, two cells filled 5 m each
    r = cut_fill_balance(z, platform_elev=5.0)
    assert r["cut"] == pytest.approx(10.0)
    assert r["fill"] == pytest.approx(10.0)
    assert r["net"] == pytest.approx(0.0)
    assert r["balance"] == pytest.approx(0.0)

    # platform at existing grade -> one-sided work only
    r0 = cut_fill_balance(z, platform_elev=0.0)
    assert r0["cut"] == pytest.approx(20.0)
    assert r0["fill"] == pytest.approx(0.0)

    # cell area scales the volumes
    r2 = cut_fill_balance(z, platform_elev=5.0, dx=2.0, dy=1.0)
    assert r2["cut"] == pytest.approx(20.0)
    assert r2["cell_area"] == pytest.approx(2.0)

    # no elevation given -> balanced level = area-weighted mean, |cut-fill| = 0
    opt = cut_fill_balance(z)
    assert opt["platform_elev"] == pytest.approx(5.0)
    assert opt["balance"] == pytest.approx(0.0, abs=1e-9)

    # masked platform over the high cells only: level 10, zero earthwork
    m = np.array([[False, False], [True, True]])
    mo = cut_fill_balance(z, mask=m)
    assert mo["platform_elev"] == pytest.approx(10.0)
    assert mo["cut"] == pytest.approx(0.0)
    assert mo["fill"] == pytest.approx(0.0)


def test_flow_direction_d8_steepest_descent():
    i, j = np.meshgrid(np.arange(5.0), np.arange(5.0))
    z = i + j  # rises east and south -> interior cells drain NW

    codes, sinks = flow_direction_d8(z)
    assert codes[2, 2] == 32                    # NW
    assert codes[0, 2] == 16                    # top row -> W (along ridge)
    assert codes[2, 0] == 64                    # left col -> N (along ridge)
    assert sinks[0, 0]                          # global minimum corner is a pit

    # following the D8 chain from anywhere strictly descends to a sink
    jj, ii, steps = 2, 2, 0
    while not sinks[jj, ii]:
        dj, di = D8_DELTA[codes[jj, ii]]
        assert z[jj, ii] > z[jj + dj, ii + di]
        jj, ii, steps = jj + dj, ii + di, steps + 1
        assert steps < 30
    assert sinks[jj, ii]

    # southward ramp (z = row index): uniform north flow, top row are sinks
    zs = j.astype(float)
    cs, ss = flow_direction_d8(zs)
    assert cs[2, 3] == 64                       # N
    assert ss[0, :].all()
    assert not ss[1:, :].any()
