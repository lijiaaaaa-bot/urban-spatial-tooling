"""Tests for src/closure.py — population-facility-land closure
(人口-设施-用地平衡, GB 50180-2018 千人指标)."""

from src.closure import (
    REFERENCE_INDICATORS,
    capacity_balance,
    check_population_balance,
    per_capita_balance,
    population_estimate,
    thousand_person_index,
)

# 30 ha residential at FAR 2.0 -> 600,000 m² floor -> 17,142.9 人 @ 35 m²/人.
LAND = {"residential": 300_000.0, "green": 200_000.0,
        "facilities": 100_000.0, "roads": 40_000.0}


def test_population_estimate_and_thousand_person_index():
    """Floor-area -> population, then 千人指标 -> required vs provided."""
    pop = population_estimate(300_000.0, 2.0)          # 600000 / 35
    assert abs(pop - 600_000.0 / 35.0) < 1e-9

    res = thousand_person_index(pop,
                                provided_land={"education_junior": 20_000.0})
    jr = res["rows"]["education_junior"]
    required = pop / 1000.0 * REFERENCE_INDICATORS["education_junior"]["land"]
    assert abs(jr["required"] - required) < 1e-6
    assert jr["balance"] > 0 and jr["status"] == "surplus"

    # Unprovided facilities show up as deficits, totals tie out.
    assert res["rows"]["education_primary"]["status"] == "deficit"
    assert res["rows"]["education_primary"]["balance"] < 0
    assert abs(res["total_provided"] - 20_000.0) < 1e-6
    assert res["total_balance"] < 0


def test_per_capita_balance_bands():
    """17.5 m²/人 residential < 23 -> violation; green/facilities ok."""
    res = per_capita_balance(LAND, 600_000.0 / 35.0)
    by = {r["category"]: r for r in res["rows"]}

    assert by["residential"]["status"] == "violation"
    assert by["residential"]["direction"] == "low"
    assert by["green"]["status"] == "ok"           # 11.67 >= 8
    assert by["facilities"]["status"] == "ok"      # 5.83 in [5, 8]
    assert len(res["violations"]) == 1

    # Boundary: green at exactly 8 m²/人 passes, 7.99 fails.
    tight = per_capita_balance({"residential": 300_000.0,
                                "green": 8.0 * (600_000.0 / 35.0),
                                "facilities": 100_000.0},
                               600_000.0 / 35.0)
    assert {r["category"]: r["status"] for r in tight["rows"]} == \
        {"residential": "violation", "green": "ok", "facilities": "ok"}


def test_capacity_balance_and_orchestrator():
    """Over/under/balanced zones, then the combined PASS/REVISE verdict."""
    zones = [
        {"name": "Z1", "land": 150_000.0, "far": 2.0, "planned": 10_000.0},
        {"name": "Z2", "land": 150_000.0, "far": 2.0, "planned": 7_000.0},
        {"name": "Z3", "land": 140_000.0, "far": 2.0, "planned": 8_000.0},
    ]
    res = capacity_balance(zones)
    assert res["zones"]["Z1"]["verdict"] == "over_density"   # ratio 1.17
    assert res["zones"]["Z2"]["verdict"] == "under_density"  # ratio 0.82
    assert res["zones"]["Z3"]["verdict"] == "balanced"       # ratio 1.00

    # Land with residential per-capita below the 23-38 band -> REVISE.
    verdict = check_population_balance(
        LAND, zones, provided_land={"education_primary": 40_000.0})
    assert set(verdict) >= {"method1", "method2", "method3",
                            "findings", "verdict"}
    assert verdict["verdict"] == "REVISE"
    assert any("M2 居住用地" in f for f in verdict["findings"])

    # A fully compliant scheme (per-capita in band, capacities matched)
    # passes: 40 ha at FAR 1.5 -> 600,000 m² floor, planned = capacity.
    ok_land = {"residential": 400_000.0, "green": 200_000.0,
               "facilities": 120_000.0, "roads": 60_000.0}
    cap = 200_000.0 * 1.5 / 35.0
    ok_zones = [{"name": "A", "land": 200_000.0, "far": 1.5, "planned": cap},
                {"name": "B", "land": 200_000.0, "far": 1.5, "planned": cap}]
    ok_provided = {"education_junior": 17_000.0, "education_primary": 35_000.0,
                   "education_kindergarten": 12_000.0, "health": 1_200.0,
                   "elderly": 2_200.0, "culture": 2_200.0, "sports": 3_600.0,
                   "commercial": 7_200.0, "community": 4_000.0}
    ok = check_population_balance(ok_land, ok_zones,
                                  provided_land=ok_provided)
    assert ok["verdict"] == "PASS"
    assert ok["findings"] == []
