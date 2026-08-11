"""Tests for src/ventilation.py — ventilation / wind corridor analysis."""

import numpy as np
import pytest
from shapely.geometry import box

from src.ventilation import (
    wind_speed_reduction,
    ventilation_ratio,
    frontal_area_density,
    classify_ventilation,
    ventilation_corridors,
    windward_leeward,
    frontal_width,
)


def test_wind_speed_reduction_formula_and_clamp():
    # exact formula: V_ref * (1 - C_d H/W), C_d=1 -> H=20, W=40 halves the speed
    assert wind_speed_reduction(20, 40, C_d=1.0, V_ref=5.0) == pytest.approx(2.5)
    # clamp: H/W beyond 1/C_d never goes negative, floors at 0.05
    assert wind_speed_reduction(100, 10, C_d=1.2) == pytest.approx(0.05)
    # monotonicity: taller buildings / narrower streets reduce the ratio
    assert wind_speed_reduction(10, 50) > wind_speed_reduction(20, 50)
    assert wind_speed_reduction(20, 40) < wind_speed_reduction(20, 80)
    # ventilation_ratio is the V_ref=1 special case
    assert ventilation_ratio(20, 40, C_d=1.0) == pytest.approx(0.5)
    # open street (no flanking buildings) is untouched
    assert ventilation_ratio(0.0, 60.0) == pytest.approx(1.0)


def test_frontal_area_density_single_building_direction():
    # 20 x 30 m building, 20 m tall, in a 100 x 100 m cell
    b = box(0, 0, 20, 30)
    lf, xs, ys = frontal_area_density([b], [20.0], wind_from_az=270.0,  # west wind
                                      cell=100.0, domain=(0, 0, 100, 100))
    # west wind (blowing east): frontal wall = the 30 m side
    assert lf[0, 0] == pytest.approx(30.0 * 20.0 / 10000.0)  # 0.06
    # south wind (blowing north): frontal wall = the 20 m side
    lf2, _, _ = frontal_area_density([b], [20.0], wind_from_az=0.0,
                                     cell=100.0, domain=(0, 0, 100, 100))
    assert lf2[0, 0] == pytest.approx(20.0 * 20.0 / 10000.0)  # 0.04
    # NW wind (blowing SE): frontal width = (20+30) * cos(45 deg)
    lf3, _, _ = frontal_area_density([b], [20.0], wind_from_az=315.0,
                                     cell=100.0, domain=(0, 0, 100, 100))
    assert lf3[0, 0] == pytest.approx((20.0 + 30.0) * np.cos(np.pi / 4) * 20.0 / 10000.0)
    # frontal_width agrees with the perpendicular-silhouette definition
    bhat = np.array([0.70710678, -0.70710678])  # NW blowing toward SE
    c45 = np.cos(np.pi / 4)
    assert frontal_width(b, bhat) == pytest.approx(20.0 * c45 + 30.0 * c45)


def test_classify_ventilation_thresholds():
    lf = np.array([[0.2, 0.3, 0.5, 0.7]])
    cls = classify_ventilation(lf)
    # < 0.3 good (0); 0.3-0.5 medium (1); > 0.5 poor (2)
    assert cls[0, 0] == 0
    assert cls[0, 1] == 1 and cls[0, 2] == 1
    assert cls[0, 3] == 2


def test_ventilation_corridors_channel_labeling():
    # 5 x 5 grid, one open column -> one corridor component of 5 cells
    lf = np.ones((5, 5)) * 1.0
    lf[:, 2] = 0.1                       # open column
    labels, n = ventilation_corridors(lf)
    assert n == 1
    assert labels[:, 2].tolist() == [1] * 5
    assert (labels[:, 0] == 0).all() and (labels[:, 1] == 0).all()

    # 8-connectivity: diagonally touching open cells merge into one component
    lf2 = np.ones((3, 3)) * 1.0
    lf2[0, 0] = 0.1
    lf2[1, 1] = 0.1
    labels2, n2 = ventilation_corridors(lf2)
    assert n2 == 1 and labels2[0, 0] == labels2[1, 1] != 0

    # two separate open clusters stay separate
    lf3 = np.ones((4, 4)) * 1.0
    lf3[0, 0] = 0.1
    lf3[3, 3] = 0.1
    labels3, n3 = ventilation_corridors(lf3)
    assert n3 == 2
    assert labels3[0, 0] != labels3[3, 3]


def test_windward_leeward_corridor_throughflow():
    # an open column + wind from NW must produce a windward->leeward
    # through-corridor (the planning-relevant property)
    lf = np.ones((5, 5)) * 1.0
    lf[:, 2] = 0.1                       # open column
    labels, n = ventilation_corridors(lf)
    ww, ll = windward_leeward(labels, wind_from_az=315.0)  # NW wind
    m = labels == 1
    assert m[ww].any() and m[ll].any()   # touches both the NW and SE edges

    # wind from the east: open column is now parallel to the wind and does
    # NOT span windward (E) to leeward (W)
    ww2, ll2 = windward_leeward(labels, wind_from_az=90.0)
    assert not (m[ww2].any() and m[ll2].any())
