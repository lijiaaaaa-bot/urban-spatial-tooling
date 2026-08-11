"""Population-facility-land closure (人口-设施-用地平衡).

Extracted from notebooks/20-population-facility-balance.ipynb (target B-04).

The closure chain of residential planning is

    population = 住宅建筑面积 / 人均建筑面积
               = 居住用地 x FAR / 人均建筑面积
               = 居住用地 / 人均居住用地

Three methods verify that population, facility provision and land
allocation stay consistent with each other, plus one orchestrator:

- ``population_estimate``   — Method 1 step 1: population estimate from
  residential land x FAR (per-capita floor area, default 35 m²/人).
- ``thousand_person_index`` — Method 1 step 2: population x 千人指标
  -> required facility floor / land per facility type, compared with the
  provided land allocation -> surplus / deficit per type.
- ``per_capita_balance``   — Method 2: actual per-capita land (from a
  land-use GeoJSON area summary) vs reference bands: residential
  23-38 m²/人, green >= 8 m²/人, public facilities 5-8 m²/人.
- ``capacity_balance``     — Method 3: maximum population capacity per
  zone (floor area / per-capita floor area) vs planned population ->
  over-density / under-density / balanced zones.
- ``check_population_balance`` — orchestrator: runs all three methods on
  one land summary and returns a combined PASS / REVISE verdict.

Standard references
-------------------
- GB 50180-2018《城市居住区规划设计标准》表5.0.3 配套设施控制指标
  (通称"千人指标"; 旧版 GB 50180-93 中位于附录A, 配套设置要求见附录B).
- 条文说明 4.0.2/4.0.3: 各项用地指标按人均住房建筑面积 35 m² 测算
  -> PER_CAPITA_FLOOR = 35.
- 表5.0.3 的分圈层嵌套口径 (十五分钟 ⊃ 十分钟 ⊃ 五分钟, 指标不累计).
  本模块按"全居住区一表"工程简化; REFERENCE_INDICATORS 为演示用参考值
  (依据条文说明中的学校规模、派出所 32~40 m²/千人 等锚点推导), 正式
  项目应以标准原文数值分圈层叠加, 并写入 assumptions.json.

Parameter sensitivity: all thresholds are module-level constants (C-type,
parameter-sensitive analysis). Pure python — no GPU, no geopandas.
"""

from typing import Dict, List, Optional, Sequence

# ═══════════════════════════════════════════════════════════════
# Reference constants (C-type parameters — calibrate per site)
# ═══════════════════════════════════════════════════════════════

#: 人均住房建筑面积 (m²/人) — GB 50180-2018 条文说明 4.0.2/4.0.3 测算基准.
PER_CAPITA_FLOOR: float = 35.0
#: 人均居住用地参考带 (m²/人).
RESIDENTIAL_BAND: tuple = (23.0, 38.0)
#: 人均绿地下限 (m²/人).
GREEN_MIN: float = 8.0
#: 人均公共服务设施用地参考带 (m²/人, 口径宽于千人指标下限, 含室外场地).
FACILITY_BAND: tuple = (5.0, 8.0)
#: 规划/容量比值容差 (|ratio - 1| <= RATIO_TOLERANCE 视为均衡).
RATIO_TOLERANCE: float = 0.10

#: 演示用参考千人指标 (m²/千人, 全居住区口径) — 用地/建筑面积.
#: 锚点: 条文说明 5.0.3 学校规模 (十五分钟圈 2x24班~1x24+2x36班 初中,
#: 十分钟圈 36~54班 小学, 五分钟圈 12~18班 幼儿园) + 生均用地 ~20-22 m²;
#: 派出所 32~40 m²/千人. 精确数值以 GB 50180-2018 表5.0.3 原文为准.
REFERENCE_INDICATORS: Dict[str, Dict[str, float]] = {
    "education_junior":       {"zh": "教育·初中", "land": 950.0, "floor": 640.0},
    "education_primary":      {"zh": "教育·小学", "land": 1950.0, "floor": 1320.0},
    "education_kindergarten": {"zh": "教育·幼儿园", "land": 650.0, "floor": 440.0},
    "health":                 {"zh": "医疗卫生", "land": 60.0, "floor": 60.0},
    "elderly":                {"zh": "养老设施", "land": 120.0, "floor": 100.0},
    "culture":                {"zh": "文化设施", "land": 120.0, "floor": 110.0},
    "sports":                 {"zh": "体育设施", "land": 200.0, "floor": 60.0},
    "commercial":             {"zh": "商业服务", "land": 400.0, "floor": 400.0},
    "community":              {"zh": "社区服务", "land": 220.0, "floor": 200.0},
}

# ═══════════════════════════════════════════════════════════════
# Method 1 — 千人指标 cross-check 千人指标交叉校验
# ═══════════════════════════════════════════════════════════════

def population_estimate(
    residential_land_m2: float,
    far: float,
    per_capita_floor: float = PER_CAPITA_FLOOR,
) -> float:
    """Method 1 step 1 — resident population from residential floor area.

    population = residential_land x FAR / per_capita_floor.  When zones
    have different FARs, pass the land-weighted average FAR.

    Args:
        residential_land_m2: residential land area (m², projected CRS).
        far: floor-area ratio (总建筑面积 / 用地面积).
        per_capita_floor: m² housing floor area per capita (default 35).

    Returns:
        Estimated population (people).
    """
    floor_area = residential_land_m2 * far
    return floor_area / per_capita_floor


def thousand_person_index(
    population: float,
    indicators: Optional[Dict[str, dict]] = None,
    provided_land: Optional[Dict[str, float]] = None,
) -> dict:
    """Method 1 step 2 — required facility floor/land vs provided land.

    千人指标: required = population / 1000 x indicator (m²/千人).

    Args:
        population: population estimate from :func:`population_estimate`.
        indicators: {category: {"zh", "land", "floor"}} per-thousand-person
            indicators; defaults to REFERENCE_INDICATORS.
        provided_land: {category: m²} actually allocated facility land.

    Returns:
        {"rows": {category: {"zh", "required", "provided", "balance",
         "status"}}, "total_required", "total_provided", "total_balance",
         "population"}; status is "ok" (within +-1%), "surplus" or
         "deficit".
    """
    indicators = indicators if indicators is not None else REFERENCE_INDICATORS
    provided_land = provided_land if provided_land is not None else {}
    units = population / 1000.0
    rows: Dict[str, dict] = {}
    total_required = total_provided = total_balance = 0.0
    for key, spec in indicators.items():
        required = units * spec["land"]
        provided = float(provided_land.get(key, 0.0))
        balance = provided - required
        if abs(balance) <= 0.01 * required:      # +-1% 视为均衡
            status = "ok"
        else:
            status = "surplus" if balance > 0 else "deficit"
        rows[key] = {"zh": spec["zh"], "required": required,
                     "provided": provided, "balance": balance,
                     "status": status}
        total_required += required
        total_provided += provided
        total_balance += balance
    return {"rows": rows, "population": population,
            "total_required": total_required,
            "total_provided": total_provided,
            "total_balance": total_balance}


# ═══════════════════════════════════════════════════════════════
# Method 2 — Per-capita land balance 人均用地平衡
# ═══════════════════════════════════════════════════════════════

def per_capita_balance(
    land_areas: Dict[str, float],
    population: float,
    bands: Optional[Dict[str, tuple]] = None,
) -> dict:
    """Method 2 — per-capita land against GB 50180-2018 reference bands.

    Actual per-capita land = land_areas[category] / population, compared
    with: residential 23-38 m²/人, green >= 8 m²/人, public facilities
    5-8 m²/人 (roads are informational and not band-checked).

    Args:
        land_areas: {category: m²} aggregated from a land-use GeoJSON
            (categories "residential", "green", "facilities", "roads").
        population: population basis (use the Method-1 estimate so the
            three methods close on the same population).
        bands: optional override of the {category: (low, high)} bands;
            None high means "no upper bound".

    Returns:
        {"rows": [{"category", "zh", "per_capita", "band", "status",
         "direction"}], "violations": [str]}; status "ok" | "violation".
    """
    bands = bands or {"residential": RESIDENTIAL_BAND,
                      "green": (GREEN_MIN, None),
                      "facilities": FACILITY_BAND}
    zh = {"residential": "居住用地", "green": "绿地", "facilities": "公共服务设施"}
    rows: List[dict] = []
    violations: List[str] = []
    for category, (low, high) in bands.items():
        per_capita = land_areas.get(category, 0.0) / population
        direction = None
        if low is not None and per_capita < low:
            direction = "low"
        elif high is not None and per_capita > high:
            direction = "high"
        status = "violation" if direction else "ok"
        if low is None:
            band = f">={high}"
        elif high is None:
            band = f"{low}~∞"
        else:
            band = f"{low}~{high}"
        rows.append({"category": category, "zh": zh[category],
                     "per_capita": per_capita, "band": band,
                     "status": status, "direction": direction})
        if direction:
            lo_hi = "低于下限" if direction == "low" else "高于上限"
            violations.append(
                f"{zh[category]} {per_capita:.1f} m²/人 {lo_hi} {band} m²/人")
    return {"rows": rows, "violations": violations}


# ═══════════════════════════════════════════════════════════════
# Method 3 — Population capacity vs planned 人口容量校验
# ═══════════════════════════════════════════════════════════════

def capacity_balance(
    zones: Sequence[dict],
    per_capita_floor: float = PER_CAPITA_FLOOR,
) -> dict:
    """Method 3 — zone population capacity vs planned population.

    Per zone: capacity = land x far / per_capita_floor; ratio =
    planned / capacity.  ratio > 1 + tolerance -> "over_density"
    (计划人口超出住房容量), ratio < 1 - tolerance -> "under_density"
    (容量富余), otherwise "balanced".

    Args:
        zones: [{"name", "land": m², "far", "planned": people}].
        per_capita_floor: m² housing floor area per capita.

    Returns:
        {"zones": {name: {"land", "far", "floor", "capacity", "planned",
         "ratio", "verdict"}}, "total_capacity", "total_planned",
         "total_ratio"}.
    """
    zones_out: Dict[str, dict] = {}
    total_capacity = total_planned = 0.0
    for z in zones:
        floor = z["land"] * z["far"]
        capacity = floor / per_capita_floor
        planned = float(z["planned"])
        ratio = planned / capacity if capacity > 0 else float("inf")
        if abs(ratio - 1.0) > RATIO_TOLERANCE:
            verdict = "over_density" if ratio > 1.0 else "under_density"
        else:
            verdict = "balanced"
        zones_out[z["name"]] = {"land": z["land"], "far": z["far"],
                                "floor": floor, "capacity": capacity,
                                "planned": planned, "ratio": ratio,
                                "verdict": verdict}
        total_capacity += capacity
        total_planned += planned
    return {"zones": zones_out, "total_capacity": total_capacity,
            "total_planned": total_planned,
            "total_ratio": total_planned / total_capacity
            if total_capacity > 0 else float("inf")}


# ═══════════════════════════════════════════════════════════════
# Orchestrator — 三种方法闭合校验 (target B-04 check function)
# ═══════════════════════════════════════════════════════════════

def check_population_balance(
    land_areas: Dict[str, float],
    zones: Sequence[dict],
    indicators: Optional[Dict[str, dict]] = None,
    provided_land: Optional[Dict[str, float]] = None,
    planned_population: Optional[float] = None,
) -> dict:
    """Run all three closure methods and return a combined verdict.

    Population basis = Method-1 estimate (residential land x weighted
    FAR / 35 m²/人), so Method 1/2/3 close on the same population.

    Args:
        land_areas: {category: m²} from the land-use GeoJSON.
        zones: Method-3 zone descriptors (see :func:`capacity_balance`).
        indicators: Method-1 per-thousand-person indicators.
        provided_land: Method-1 provided facility land (m²).
        planned_population: planned population; defaults to the zone sum.

    Returns:
        {"method1", "method2", "method3", "findings": [str] (issues),
         "notes": [str] (informational surpluses),
         "verdict": "PASS" | "REVISE"}.
    """
    total_land = sum(z["land"] for z in zones)
    avg_far = sum(z["land"] * z["far"] for z in zones) / total_land \
        if total_land > 0 else 1.0
    population = population_estimate(land_areas.get("residential", 0.0),
                                     avg_far)
    m1 = thousand_person_index(population, indicators, provided_land)
    m2 = per_capita_balance(land_areas, population)
    m3 = capacity_balance(zones)

    planned = (planned_population if planned_population is not None
               else m3["total_planned"])
    findings: List[str] = []
    notes: List[str] = []
    for key, row in m1["rows"].items():
        if row["status"] == "deficit":
            findings.append(f"M1 {row['zh']}: 缺口 {row['balance']:,.0f} m²")
        elif row["status"] == "surplus":
            notes.append(f"M1 {row['zh']}: 富余 {row['balance']:+,.0f} m²")
    findings.extend(f"M2 {v}" for v in m2["violations"])
    for name, z in m3["zones"].items():
        if z["verdict"] != "balanced":
            findings.append(
                f"M3 {name}: {z['verdict']} (规划/容量 = {z['ratio']:.2f})")
    verdict = "REVISE" if findings else "PASS"
    return {"method1": m1, "method2": m2, "method3": m3,
            "planned_population": planned, "estimated_population": population,
            "findings": findings, "notes": notes, "verdict": verdict}
