"""
Urban Spatial Tooling — exploration package.

Reusable modules for urban spatial generation, analysis, and visualization.
Extracted from the notebooks under ``notebooks/``; each module is
standalone, deterministic, and CPU-only.

Modules
-------
projection   — CRS transforms (EPSG:4326 <-> EPSG:4548) and area
               computation (notebook 01)
generation   — land-use subdivision, building footprint and road
               generation (notebooks 02/03/04)
topology     — coverage / overlap / containment checks (notebook 05)
visualization — professional planning-figure rendering (notebook 07)
solar        — solar geometry, spacing coefficient, insolation grid;
               GB 50180-2018 大寒日 daylight standard (notebook 10)
compliance   — view-corridor height envelopes (三山五园) and three-lines
               (三区三线) parcel checks (notebooks 11/15)
sponge       — 海绵城市 容积法 storage volume and LID sizing;
               DB11/685-2021 >= 85% control rate (notebook 12)
living_circle— 15-minute living circle evaluation, TD/T 1062-2021
               (notebook 13)
setback      — building setback envelopes (建筑退线), footprint
               generation, and inter-building fire separation
               (建筑防火间距, GB 50016-2014) (notebook 14)
pipeline     — technical-review AND-gate: CheckResult / Severity /
               Category / TechnicalReviewRunner (notebook 16)
adapter      — haidian-format GeoJSON features -> pipeline check inputs:
               height/spacing/setback estimation, land-use code mapping,
               submission geometry/ loader
integration  — end-to-end technical review of haidian submissions:
               load -> adapt -> TechnicalReviewRunner -> tech_review.json
ventilation  — ventilation / wind corridor analysis 通风分析: empirical
               wake model, frontal area density, corridor components;
               Beijing 通风廊道 guidance (notebook 17)
renewal      — urban renewal 留-改-拆 classification: rule scoring,
               spatial clustering (block coherence), FAR-capacity
               assessment; 北京市城市更新条例 (2022) (notebook 18)
traffic      — traffic capacity analysis 交通承载力: road V/C saturation,
               intersection LOS A-F, network-wide saturation and road
               density check; GB/T 51328-2018, CJJ 37-2012 (notebook 19)
vertical     — vertical design / grading 竖向设计: slope & aspect from
               numpy gradients, cut-fill balance for a platform level,
               D8 drainage direction / sinks; CJJ 83-2016 slope classes
               (notebook 21, data-gated — synthetic DEM demo)

Archived experimental GPU benchmarks (not for production use; see
src/experimental/README.md):
experimental.metal_projection — Metal MPS projection benchmark (notebook 08)
experimental.metal_spatial    — Metal MPS spatial ops benchmark (notebook 09)
"""
