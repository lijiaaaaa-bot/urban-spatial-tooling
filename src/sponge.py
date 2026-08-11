"""
Sponge-city stormwater volume calculations (海绵城市容积法).

Extracted from notebooks/12-sponge-city.ipynb.

Provides:
- Design rainfall lookup for Beijing annual runoff control rates
  (design_rainfall_mm, H_CTRL)
- required_storage_m3: the 容积法 formula V = 10 x H x phi x F
- Runoff coefficient tables per surface type and land-use code
- LID facility sizing: lid_area = V / h_eff plus the LID_CATALOG
- sponge_city_check: the CODE-step prototype that takes a land-use
  GeoDataFrame and returns storage metrics + a LID allocation plan
- Continuous simulation (SWMM-style lumped model): chicago_hyetograph
  (Keifer-Chu temporal distribution on the Beijing storm intensity
  formula), horton_infiltration_capacity_mm_h, water_balance_step and
  continuous_simulation — upgrade from a single design rainfall depth
  to a 5-min time-step rainfall -> infiltration -> storage -> runoff
  water balance, plus LID performance curves

Standard references
-------------------
- 国办发〔2015〕75号 — national sponge-city guidance (70% of urban
  rainfall captured and reused on site).
- GB 50014-2021《室外排水设计标准》— 容积法 formula and runoff
  coefficients (径流系数).
- DB11/685-2021《海绵城市雨水控制与利用工程设计规范》(北京) — the
  governing standard for Beijing: **新建项目年径流总量控制率 ≥ 85%**.
  (The older DB11/685-2013 value of 75% does not govern new projects.)
  Design rainfall depth at 85%: H = 33.6 mm.
- Parcel-level "3-5-7" indicators (DB11/685-2021, 新建): 下凹绿地率
  ≥ 50%, 透水铺装率 ≥ 70%, 调蓄容积 ≥ 30 m³/1000 m² — these bind the
  *placement* of LID within each parcel, not just the site-wide total.

Parameter sensitivity: the runoff coefficient phi dominates the result
(parcel-weighted phi ~0.53 vs tabulated 0.60 differ by +13.5% in volume).
Always document the phi assumption in the output.

All computations are pure numpy + geopandas — no GPU dependency.
"""

from typing import Dict, Optional

import geopandas as gpd
import numpy as np

try:
    from .projection import compute_area_4548
except ImportError:
    from projection import compute_area_4548

# Beijing design rainfall depth (mm) by annual runoff control rate (%)
# From the 技术指南 control-rate lookup table (long-run daily rainfall
# statistics).  85% is the new-development design point (DB11/685-2021).
H_CTRL: Dict[int, float] = {70: 17.2, 75: 21.4, 80: 26.7, 85: 33.6}

# Design rainfall depth (mm) for 24 h drainage return periods (Beijing).
# Peak-control sizing uses the design storm; exact values come from the
# local IDF curve (暴雨强度公式).  Values below are representative.
H_STORM: Dict[int, float] = {1: 45, 2: 55, 5: 70, 10: 85, 20: 100}

# Runoff coefficients per surface type (GB 50014-2021 ranges)
PHI_SURFACE: Dict[str, float] = {
    'roof': 0.90,                 # 屋面
    'impervious_road': 0.90,      # 混凝土/沥青路面
    'permeable_pavement': 0.35,   # 透水铺装 (post-LID, 技术指南 0.29-0.36)
    'green': 0.15,                # 绿地
    'water': 1.00,                # 水面
}

# Composite runoff coefficient per land-use code — a parcel's phi is the
# area-weighted mix of its surfaces (roof + road + green inside the parcel)
PHI_BY_LANDUSE: Dict[str, float] = {
    'R': 0.55,  # 居住
    'A': 0.60,  # 公共管理与公共服务
    'B': 0.65,  # 商业服务业
    'G': 0.15,  # 绿地与广场
    'S': 0.85,  # 道路与交通设施
    'E': 0.50,  # 教育/其他
}

# Default land-use mix used by the generation pipeline (src/generation.py)
LAND_USE_MIX: Dict[str, float] = {'R': 0.35, 'A': 0.10, 'B': 0.12,
                                  'G': 0.20, 'S': 0.15, 'E': 0.08}

# LID facility catalog: effective storage depth h_eff (m) = surface
# ponding depth + media voids (porosity x media thickness), typical values
# from the 技术指南 / DB11 685.
LID_CATALOG: Dict[str, dict] = {
    'permeable_pavement': dict(h_eff=0.105, note='透水铺装: 0.30 m 碎石层, 孔隙率 0.35'),
    'rain_garden':        dict(h_eff=0.250, note='雨水花园: 0.15 m 蓄水层 + 0.60 m 种植土'),
    'sunken_green':       dict(h_eff=0.150, note='下沉式绿地: 0.15 m 下凹深度'),
    'bioretention':       dict(h_eff=0.300, note='生物滞留设施: 0.20 m 蓄水层 + 0.60 m 滤料'),
}

# Typical design split of storage across LID types
DEFAULT_ALLOC: Dict[str, float] = {
    'rain_garden': 0.40, 'permeable_pavement': 0.30,
    'sunken_green': 0.20, 'underground_tank': 0.10,
}

# Beijing new-development governing control rate (DB11/685-2021)
DESIGN_CONTROL_RATE = 0.85

# ── Continuous simulation parameters ────────────────────────────────

# Simplified Beijing storm intensity formula (暴雨强度公式):
#     i(t) = a / (t + b)^n     i in mm/min, t in minutes
# with the Chicago (Keifer-Chu) peak ratio r.  (The full Beijing formula
# carries an extra 1 + C·lgP frequency term; the simplified form here is
# used to shape the hyetograph, with the storm depth supplied separately.)
BEIJING_IDF: dict = dict(a=8.0, b=0.7, n=0.6, r=0.4)

# Horton infiltration parameters for urban loam (classic values from
# Horton's original example): f0 initial capacity (mm/h), fc steady
# capacity (mm/h), k decay constant (1/h).
HORTON_LOAM: dict = dict(f0=76.0, fc=13.0, k=4.2)

# Controlled release rate from LID storage to the drainage network (mm/h
# over the catchment) — underdrain + green drainage, typical design range
# 2-10 mm/h.
DEFAULT_RELEASE_MM_H = 5.0


def chicago_hyetograph(duration_min: float = 1440.0, dt_min: float = 5.0,
                       P_total_mm: float = 45.0, r: Optional[float] = None,
                       a: Optional[float] = None, b: Optional[float] = None,
                       n: Optional[float] = None,
                       t_peak_min: Optional[float] = None) -> dict:
    """Chicago-method design-storm hyetograph (Keifer-Chu 1957).

    Applies the Keifer-Chu temporal distribution to the Beijing storm
    intensity formula i(t) = a / (t + b)^n (mm/min, t in min): the storm
    of total duration ``duration_min`` is split at the peak time
    t_p = r · T, and each limb is built from the derivative of the
    cumulative depth H(u) = a·u / (u + b)^n evaluated at the scaled
    partial duration u:

        rising limb  (t <= t_p): u = (t_p - t) / r
        falling limb (t >  t_p): u = (t - t_p) / (1 - r)

    Step depths are rescaled so the total depth equals ``P_total_mm``
    (the design depth of the chosen return period, e.g. from
    :data:`H_STORM`).

    Parameters
    ----------
    duration_min : float
        Storm duration (min); default 24 h.
    dt_min : float
        Time step (min); default 5 min.
    P_total_mm : float
        Total rainfall depth of the design storm (mm).
    r : float, optional
        Peak ratio (time-to-peak / duration); default Beijing 0.4
        (:data:`BEIJING_IDF`).
    a, b, n : float, optional
        Storm intensity formula parameters; default Beijing
        a=8.0, b=0.7, n=0.6 (:data:`BEIJING_IDF`).
    t_peak_min : float, optional
        Explicit peak time; defaults to r · duration_min.

    Returns
    -------
    dict
        ``times_min`` (step start times), ``depth_mm`` (rainfall depth
        per step), ``intensity_mm_min``, ``cumulative_mm`` (depth at
        step end), ``dt_min``, ``duration_min``, ``t_peak_min``,
        ``P_total_mm``, ``peak_intensity_mm_min`` and the parameters
        used.
    """
    if r is None:
        r = BEIJING_IDF['r']
    if a is None:
        a = BEIJING_IDF['a']
    if b is None:
        b = BEIJING_IDF['b']
    if n is None:
        n = BEIJING_IDF['n']
    if not 0.0 < r < 1.0:
        raise ValueError(f'peak ratio r must be in (0, 1), got {r}')
    if t_peak_min is None:
        t_peak_min = r * duration_min
    if not 0.0 < t_peak_min < duration_min:
        raise ValueError(f't_peak_min {t_peak_min} outside (0, {duration_min})')

    n_steps = int(np.ceil(duration_min / dt_min))
    times = np.arange(n_steps) * dt_min
    depths = np.empty(n_steps)
    for k, t0 in enumerate(times):
        t1 = min(t0 + dt_min, duration_min)
        if t1 <= t_peak_min:
            # rising limb, u decreases from t_p/r to 0
            depths[k] = _idf_seg_depth((t_peak_min - t0) / r,
                                       (t_peak_min - t1) / r, a, b, n)
        elif t0 >= t_peak_min:
            # falling limb, v increases from 0 to (T - t_p)/(1-r)
            depths[k] = _idf_seg_depth((t1 - t_peak_min) / (1.0 - r),
                                       (t0 - t_peak_min) / (1.0 - r), a, b, n)
        else:
            # step straddles the peak: sum the two partial depths
            depths[k] = (_idf_seg_depth((t_peak_min - t0) / r, 0.0, a, b, n)
                         + _idf_seg_depth((t1 - t_peak_min) / (1.0 - r),
                                          0.0, a, b, n))
    depths = depths * (P_total_mm / depths.sum())

    return {
        'times_min': times,
        'depth_mm': depths,
        'intensity_mm_min': depths / dt_min,
        'cumulative_mm': np.cumsum(depths),
        'dt_min': dt_min,
        'duration_min': duration_min,
        't_peak_min': t_peak_min,
        'r': r, 'a': a, 'b': b, 'n': n,
        'P_total_mm': float(depths.sum()),
        'peak_intensity_mm_min': float(depths.max() / dt_min),
    }


def _idf_seg_depth(u_hi: float, u_lo: float, a: float, b: float,
                   n: float) -> float:
    """Rainfall depth (mm) of an IDF partial duration segment.

    H(u) = a·u / (u + b)^n is the cumulative depth for IDF duration u
    (min); the segment depth is H(u_hi) - H(u_lo).
    """
    def _cum(u: float) -> float:
        return a * u / (u + b) ** n

    return _cum(u_hi) - _cum(u_lo)


def horton_infiltration_capacity_mm_h(t_hours,
                                      f0: Optional[float] = None,
                                      fc: Optional[float] = None,
                                      k: Optional[float] = None):
    """Horton infiltration capacity (mm/h) at elapsed wet time t (h).

    f(t) = fc + (f0 - fc) · e^(-k·t); vectorized over ``t_hours``.
    Defaults to urban-loam values :data:`HORTON_LOAM`
    (f0=76, fc=13 mm/h, k=4.2 1/h).
    """
    if f0 is None:
        f0 = HORTON_LOAM['f0']
    if fc is None:
        fc = HORTON_LOAM['fc']
    if k is None:
        k = HORTON_LOAM['k']
    t = np.asarray(t_hours, dtype=float)
    return fc + (f0 - fc) * np.exp(-k * t)


def water_balance_step(S_mm: float, rainfall_mm: float, dt_min: float,
                       phi: float, f_inf_mm: float = 0.0,
                       S_cap_mm: Optional[float] = None,
                       q_release_mm_h: float = 0.0):
    """One time step of the lumped water balance.

    Units are mm of water over the catchment (m³ = mm · 10 · F_ha).
    Per-step sequence (documented for reproducibility):

    1. rainfall is split by the runoff coefficient phi: the impervious
       share phi·r drains to LID storage, the pervious share (1-phi)·r
       infiltrates;
    2. Horton infiltration consumes pervious rainfall first, then stored
       water (LID drains downward);
    3. the remaining pervious excess + impervious share enter storage,
       S(t+dt) = S(t) + inflow - infiltration;
    4. storage above capacity S_cap overflows (bypass);
    5. a controlled release q_release (mm/h) drains the storage to the
       drainage network.

    Conservation: rainfall = infiltration + runoff + ΔS.

    Parameters
    ----------
    S_mm : float
        Storage at the start of the step (mm over the catchment).
    rainfall_mm : float
        Rainfall depth this step (mm).
    dt_min : float
        Step length (min).
    phi : float
        Runoff coefficient (impervious fraction, 0-1).
    f_inf_mm : float, optional
        Horton infiltration capacity this step (mm).
    S_cap_mm : float, optional
        LID storage capacity (mm); None = unbounded.
    q_release_mm_h : float, optional
        Controlled release rate (mm/h), e.g. :data:`DEFAULT_RELEASE_MM_H`.

    Returns
    -------
    (float, dict)
        Storage at the end of the step and a dict with ``rainfall_mm``,
        ``infiltration_mm``, ``inflow_mm``, ``storage_before_mm``,
        ``storage_after_mm``, ``overflow_mm``, ``release_mm``,
        ``runoff_mm`` (overflow + release).
    """
    rain_imp = phi * rainfall_mm
    rain_perv = (1.0 - phi) * rainfall_mm

    # 1+2. infiltration: pervious rainfall first, then stored water
    inf_perv = min(f_inf_mm, rain_perv)
    inf_storage = min(max(f_inf_mm - inf_perv, 0.0), S_mm)
    infiltration = inf_perv + inf_storage

    # 3. inflow to storage and continuity
    inflow = rain_imp + (rain_perv - inf_perv)
    S1 = S_mm + inflow - inf_storage

    # 4. overflow above capacity
    overflow = 0.0
    if S_cap_mm is not None:
        overflow = max(S1 - S_cap_mm, 0.0)
        S1 -= overflow

    # 5. controlled release
    release = min(q_release_mm_h * dt_min / 60.0, S1)
    S1 -= release

    return S1, {
        'rainfall_mm': rainfall_mm,
        'infiltration_mm': infiltration,
        'inflow_mm': inflow,
        'storage_before_mm': S_mm,
        'storage_after_mm': S1,
        'overflow_mm': overflow,
        'release_mm': release,
        'runoff_mm': overflow + release,
    }


def continuous_simulation(rainfall_mm, dt_min: float = 5.0,
                          F_ha: float = 1141.28, phi: float = 0.53,
                          S_cap_m3: Optional[float] = None,
                          f0: Optional[float] = None,
                          fc: Optional[float] = None,
                          k: Optional[float] = None,
                          q_release_mm_h: float = DEFAULT_RELEASE_MM_H,
                          S0_mm: float = 0.0) -> dict:
    """SWMM-style lumped continuous simulation over a rainfall series.

    Loops :func:`water_balance_step` over every time step of the storm:
    Horton infiltration capacity decays with wet-elapsed time
    (f(t) = fc + (f0 - fc)·e^(-k·t); no dry-period recovery — a
    documented simplification), rainfall splits by phi, and LID storage
    (capacity ``S_cap_m3``, e.g. the 容积法 volume at α=85%) retains
    water that would otherwise run off, releasing it at the controlled
    rate.

    Parameters
    ----------
    rainfall_mm : array-like
        Rainfall depth per time step (mm) — e.g. the ``depth_mm``
        series from :func:`chicago_hyetograph`.
    dt_min : float
        Time step (min); default 5.
    F_ha : float
        Catchment area (ha).
    phi : float
        Runoff coefficient (0-1).
    S_cap_m3 : float, optional
        LID storage capacity (m³); None = unbounded.
    f0, fc, k : float, optional
        Horton parameters (mm/h, mm/h, 1/h); default :data:`HORTON_LOAM`.
    q_release_mm_h : float, optional
        Controlled release rate (mm/h).
    S0_mm : float, optional
        Initial storage (mm over the catchment).

    Returns
    -------
    dict
        Arrays ``time_min``, ``rainfall_mm``, ``infiltration_mm``,
        ``storage_mm``, ``overflow_mm``, ``release_mm``, ``runoff_mm``,
        ``runoff_m3_s`` plus a ``summary`` dict: total rainfall /
        infiltration / runoff / overflow / release, runoff coefficient,
        capture rate, peak inflow and runoff (m³/s), max storage (m³)
        and capacity usage.
    """
    rainfall = np.asarray(rainfall_mm, dtype=float)
    n = rainfall.size
    S_cap_mm = (None if S_cap_m3 is None
                else float(S_cap_m3) / (10.0 * F_ha))
    # Horton capacity per step (mm), evaluated at the step midpoint
    t_hours = (np.arange(n) + 0.5) * dt_min / 60.0
    f_horton = (horton_infiltration_capacity_mm_h(t_hours, f0, fc, k)
                * dt_min / 60.0)

    time_min = np.arange(n) * dt_min
    infiltration = np.empty(n)
    storage = np.empty(n)
    overflow = np.empty(n)
    release = np.empty(n)
    runoff = np.empty(n)

    S = S0_mm
    for i in range(n):
        S, step = water_balance_step(S, rainfall[i], dt_min, phi,
                                     f_inf_mm=f_horton[i],
                                     S_cap_mm=S_cap_mm,
                                     q_release_mm_h=q_release_mm_h)
        infiltration[i] = step['infiltration_mm']
        storage[i] = step['storage_after_mm']
        overflow[i] = step['overflow_mm']
        release[i] = step['release_mm']
        runoff[i] = step['runoff_mm']

    runoff_m3_s = runoff * 10.0 * F_ha / (dt_min * 60.0)
    total_rain = float(rainfall.sum())
    total_run = float(runoff.sum())
    max_storage_m3 = float(storage.max() * 10.0 * F_ha)
    summary = {
        'total_rainfall_mm': total_rain,
        'total_infiltration_mm': float(infiltration.sum()),
        'total_runoff_mm': total_run,
        'total_overflow_mm': float(overflow.sum()),
        'total_release_mm': float(release.sum()),
        'runoff_coefficient': total_run / total_rain if total_rain else 0.0,
        'capture_rate': 1.0 - (total_run / total_rain) if total_rain else 1.0,
        'peak_rainfall_mm_h': float(rainfall.max() / dt_min * 60.0),
        'peak_inflow_m3_s': (float(rainfall.max() / dt_min * 60.0)
                             * 10.0 * F_ha / 3600.0),
        'peak_runoff_m3_s': float(runoff_m3_s.max()),
        'max_storage_m3': max_storage_m3,
        'storage_used_frac': (max_storage_m3 / float(S_cap_m3)
                              if S_cap_m3 else 0.0),
    }
    return {
        'time_min': time_min,
        'rainfall_mm': rainfall,
        'infiltration_mm': infiltration,
        'storage_mm': storage,
        'overflow_mm': overflow,
        'release_mm': release,
        'runoff_mm': runoff,
        'runoff_m3_s': runoff_m3_s,
        'F_ha': F_ha,
        'phi': phi,
        'S_cap_m3': S_cap_m3,
        'summary': summary,
    }


__all__ = [
    "H_CTRL", "H_STORM", "PHI_SURFACE", "PHI_BY_LANDUSE", "LAND_USE_MIX",
    "LID_CATALOG", "DEFAULT_ALLOC", "DESIGN_CONTROL_RATE",
    "BEIJING_IDF", "HORTON_LOAM", "DEFAULT_RELEASE_MM_H",
    "design_rainfall_mm", "composite_phi_by_mix", "required_storage_m3",
    "lid_area", "add_phi_and_storage", "sponge_city_check",
    "chicago_hyetograph", "horton_infiltration_capacity_mm_h",
    "water_balance_step", "continuous_simulation",
]


def design_rainfall_mm(control_rate: float) -> float:
    """Design rainfall depth (mm) for an annual runoff control rate.

    Looks up the nearest entry in :data:`H_CTRL` (clamped to the
    supported 70-85% range).

    Parameters
    ----------
    control_rate : float
        Annual runoff volume control rate, 0.70-0.85 (Beijing new
        development: 0.85 per DB11/685-2021).

    Returns
    -------
    float
        Design rainfall depth in mm.
    """
    pct = control_rate * 100.0
    key = min(H_CTRL, key=lambda k: abs(k - pct))
    return H_CTRL[key]


def composite_phi_by_mix(mix: Optional[Dict[str, float]] = None) -> float:
    """Composite runoff coefficient from a land-use mix.

    Parameters
    ----------
    mix : dict, optional
        {land_use_code: share}, defaulting to :data:`LAND_USE_MIX`.

    Returns
    -------
    float
        Area-weighted composite phi.
    """
    mix = mix or LAND_USE_MIX
    return sum(mix[lu] * PHI_BY_LANDUSE[lu] for lu in mix)


def required_storage_m3(H_mm: float, phi: float, F_ha: float) -> float:
    """容积法: V = 10 x H x phi x F.

    Parameters
    ----------
    H_mm : float
        Design rainfall depth (mm).
    phi : float
        Runoff coefficient.
    F_ha : float
        Catchment area (ha).

    Returns
    -------
    float
        Required storage volume (m³).  The factor 10 converts
        (mm x ha) to m³: 1 mm of rainfall over 1 ha is 10 m³.
    """
    return 10.0 * H_mm * phi * F_ha


def lid_area(m3_storage: float, h_eff_m: float) -> float:
    """Facility surface area (m²) needed to store volume at effective depth.

    Parameters
    ----------
    m3_storage : float
        Storage volume to provide (m³).
    h_eff_m : float
        Effective storage depth of the LID facility (m).

    Returns
    -------
    float
        Required facility footprint (m²).
    """
    return m3_storage / h_eff_m


def add_phi_and_storage(gdf: gpd.GeoDataFrame, H_mm: float) -> gpd.GeoDataFrame:
    """Add per-parcel ``phi`` and ``V_m3`` columns to a land-use frame.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must have a ``land_use_code`` column; ``area_sqm`` is computed
        in EPSG:4548 if absent (see :func:`src.projection.compute_area_4548`).
    H_mm : float
        Design rainfall depth (mm).

    Returns
    -------
    GeoDataFrame
        Copy with ``phi`` and ``V_m3`` columns added.
    """
    working = gdf.copy()
    if 'area_sqm' not in working.columns:
        working = compute_area_4548(working)
    working['phi'] = working['land_use_code'].map(PHI_BY_LANDUSE)
    working['V_m3'] = (10.0 * H_mm * working['phi']
                       * working['area_sqm'] / 10_000.0)
    return working


def sponge_city_check(land_use_gdf: gpd.GeoDataFrame,
                      control_rate: float = DESIGN_CONTROL_RATE,
                      alloc: Optional[Dict[str, float]] = None) -> dict:
    """容积法 sponge-city CODE step (prototype for haidian procedure.py).

    Parameters
    ----------
    land_use_gdf : GeoDataFrame
        Land-use parcels with a ``land_use_code`` column; ``area_sqm``
        is computed in EPSG:4548 if absent.
    control_rate : float, optional
        Annual runoff volume control rate (0.70-0.85; Beijing new
        development: 0.85 per DB11/685-2021).
    alloc : dict, optional
        LID allocation fractions {facility: share}, defaulting to
        :data:`DEFAULT_ALLOC`.

    Returns
    -------
    dict
        Storage metrics + LID allocation plan:
        ``method``, ``control_rate``, ``design_rainfall_mm``,
        ``site_area_ha``, ``composite_phi``, ``required_storage_m3``,
        ``storage_density_m3_ha``, ``lid_plan``.
    """
    # 1. design rainfall depth from the Beijing control-rate lookup
    H = design_rainfall_mm(control_rate)
    # 2. per-parcel runoff with 容积法
    working = add_phi_and_storage(land_use_gdf, H)
    phi_i = working['phi']
    F_ha = working['area_sqm'] / 10_000.0
    V_total = working['V_m3'].sum()
    phi_comp = float((phi_i * F_ha).sum() / F_ha.sum())
    # 3. LID allocation
    if alloc is None:
        alloc = DEFAULT_ALLOC
    lid = {}
    for name, w in alloc.items():
        v = float(V_total * w)
        if name == 'underground_tank':
            lid[name] = {'volume_m3': v, 'area_m2': None}
        else:
            lid[name] = {'volume_m3': v,
                         'area_m2': float(v / LID_CATALOG[name]['h_eff'])}
    return {
        'method': 'volume_capture (容积法, GB 50014-2021)',
        'control_rate': control_rate,
        'design_rainfall_mm': H,
        'site_area_ha': float(F_ha.sum()),
        'composite_phi': phi_comp,
        'required_storage_m3': float(V_total),
        'storage_density_m3_ha': float(V_total / F_ha.sum()),
        'lid_plan': lid,
    }


__all__ = [
    "H_CTRL", "H_STORM", "PHI_SURFACE", "PHI_BY_LANDUSE", "LAND_USE_MIX",
    "LID_CATALOG", "DEFAULT_ALLOC", "DESIGN_CONTROL_RATE",
    "design_rainfall_mm", "composite_phi_by_mix", "required_storage_m3",
    "lid_area", "add_phi_and_storage", "sponge_city_check",
]
