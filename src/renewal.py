"""Urban renewal classification (城市更新 留-改-拆).

Extracted from notebooks/18-urban-renewal-classification.ipynb.

Provides three complementary lenses for classifying every building (or
block) of an existing site into 留 (retain) / 改 (renovate) / 拆 (demolish):

- ``classify_building``   — Method 1, rule-based per-building scoring:
  ``score = age_score + structure_score + mismatch_score``.
- ``spatial_cluster``     — Method 2, block-coherence layer: adjacent
  buildings share fate (no "独栋拆除"/tooth extraction). Takes the Method-1
  scores and the building footprints, forms an adjacency graph, and assigns
  each connected cluster its majority treatment (ties prefer 留).
- ``capacity_assessment`` — Method 3, development-value lens: the ratio of
  current FAR to planned FAR decides 拆 / 改 / 留.

Standard reference
------------------
- 《北京市城市更新条例》(2022年11月25日市十五届人大常委会第四十五次会议
  通过，2023年3月1日施行) — 城市更新分为居住类、产业类、设施类、公共
  空间类和区域综合类；坚持"留改拆"并举、以保留利用提升为主，防止大拆
  大建，采取小规模、渐进式、可持续的有机更新方式。
- 《关于在实施城市更新行动中防止大拆大建问题的通知》(建科〔2021〕63号)

Parameter sensitivity
---------------------
The regulation does NOT fix any numeric thresholds. 30 years / 15 years /
score cutoffs / FAR-ratio bands (0.5, 0.8) are reference values that
dominate the outcome distribution (C-type, parameter-sensitive analysis).
Production use must calibrate them with the planning authority and record
them in ``assumptions.json``.  All thresholds are module-level constants.

Pure numpy + shapely — no GPU, no geopandas dependency.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from shapely.geometry import Polygon

# ---------------------------------------------------------------------------
# Constants — reference thresholds (C-type sensitive; calibrate per site)
# ---------------------------------------------------------------------------

TREATMENT_ZH: Dict[str, str] = {"retain": "留", "renovate": "改", "demolish": "拆"}
#: Tie-break order for cluster majorities: 保留优先 (regulation principle).
TREATMENT_ORDER: List[str] = ["retain", "renovate", "demolish"]

AGE_OLD: int = 30      # years; age > AGE_OLD  -> demolition candidate
AGE_MID: int = 15      # years; AGE_MID..AGE_OLD -> renovation
WEAK_STRUCTURES: Tuple[str, ...] = ("masonry", "brick_wood")  # 砖混 / 砖木
SCORE_DEMOLISH: int = 5   # score >= SCORE_DEMOLISH -> 拆
SCORE_RENOVATE: int = 3   # SCORE_RENOVATE <= score < SCORE_DEMOLISH -> 改

DEMOLISH_RATIO: float = 0.5   # ratio < 0.5 * planned FAR      -> 拆
RENOVATE_RATIO: float = 0.8   # ratio < 0.8 * planned FAR      -> 改
ADJACENCY_DIST: float = 12.0  # polygon distance considered adjacent (m)
MIN_CLUSTER_SIZE: int = 3     # clusters below this size merge away


def score_to_treatment(score: float) -> str:
    """Map a Method-1 score to a treatment: 拆 / 改 / 留."""
    if score >= SCORE_DEMOLISH:
        return "demolish"
    if score >= SCORE_RENOVATE:
        return "renovate"
    return "retain"


def classify_building(
    age: float,
    structure: str,
    current_use: str,
    planned_use: str,
    unsafe: bool = False,
) -> Dict[str, Union[int, str]]:
    """Method 1 — rule-based 留-改-拆 classification of one building.

    score = age_score + structure_score + mismatch_score

    - age:       > AGE_OLD -> 3 (consider demolition); AGE_MID..AGE_OLD -> 2;
                 < AGE_MID -> 1
    - structure: masonry/brick_wood -> 2 (weak); rc -> 1 (strong)
    - mismatch:  current_use != planned_use -> 1 (needs renovation)

    ``unsafe`` (危旧房 — 鉴定危险且无修缮价值) overrides everything to
    demolition: the regulation's only unambiguous demolition case.

    Returns a dict with the three sub-scores, ``score`` and ``treatment``.
    """
    if unsafe:
        return {"age_score": 3, "structure_score": 2, "mismatch_score": 1,
                "score": 6, "treatment": "demolish"}
    age_score = 3 if age > AGE_OLD else (2 if age >= AGE_MID else 1)
    structure_score = 2 if structure in WEAK_STRUCTURES else 1
    mismatch_score = 0 if current_use == planned_use else 1
    score = age_score + structure_score + mismatch_score
    return {"age_score": age_score, "structure_score": structure_score,
            "mismatch_score": mismatch_score, "score": score,
            "treatment": score_to_treatment(score)}


def spatial_cluster(
    buildings: Sequence[Polygon],
    scores: Sequence[float],
    adjacency_dist: float = ADJACENCY_DIST,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Method 2 — adjacent buildings share fate (no 独栋拆除).

    Steps:

    1. adjacency graph: two buildings are neighbours when their polygon
       distance <= ``adjacency_dist``;
    2. connected components (BFS) -> initial clusters;
    3. clusters smaller than ``min_cluster_size`` merge into the nearest
       cluster (centroid distance), repeated until stable;
    4. each cluster gets its majority treatment; ties prefer 留
       (``TREATMENT_ORDER`` — 保留优先).

    Parameters
    ----------
    buildings : sequence of shapely Polygons (footprints)
    scores : per-building Method-1 scores (classify_building output)
    adjacency_dist : neighbour threshold in metres
    min_cluster_size : clusters below this size are merged away

    Returns
    -------
    (treatments, cluster_ids) : (n,) object array of treatments and (n,)
    int array of cluster ids — every building in a cluster shares its
    treatment by construction.
    """
    n = len(buildings)
    if n == 0:
        return np.empty(0, dtype=object), np.empty(0, dtype=int)

    # 1. adjacency graph via exact polygon distance (O(n^2) — fine for the
    #    block-scale datasets this tool targets)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = buildings[i].distance(buildings[j])
            dist[i, j] = dist[j, i] = d
    adj = dist <= adjacency_dist
    centroid = np.array([[g.centroid.x, g.centroid.y] for g in buildings])

    # 2. connected components
    cluster_id = np.full(n, -1, dtype=int)
    n_clusters = 0
    for start in range(n):
        if cluster_id[start] != -1:
            continue
        stack = [start]
        cluster_id[start] = n_clusters
        while stack:
            u = stack.pop()
            for v in np.nonzero(adj[u])[0]:
                if cluster_id[v] == -1:
                    cluster_id[v] = n_clusters
                    stack.append(v)
        n_clusters += 1

    # 3. merge under-sized clusters into the nearest neighbour (deterministic:
    #    smallest id first, nearest centroid second)
    while n_clusters > 1:
        sizes = np.bincount(cluster_id, minlength=n_clusters)
        small = [c for c in range(n_clusters) if sizes[c] < min_cluster_size]
        if not small:
            break
        target = min(small)
        cc = np.array([centroid[cluster_id == c].mean(axis=0)
                       for c in range(n_clusters)])
        others = [c for c in range(n_clusters) if c != target]
        best = min(others, key=lambda c: np.hypot(*(cc[c] - cc[target])))
        cluster_id[cluster_id == target] = best
        uniq = np.unique(cluster_id)                  # compact renumber
        remap = {old: new for new, old in enumerate(uniq)}
        cluster_id = np.array([remap[x] for x in cluster_id])
        n_clusters = len(uniq)

    # 4. majority treatment per cluster (tie -> earlier in TREATMENT_ORDER)
    per_building = np.array([score_to_treatment(s) for s in scores], dtype=object)
    final = np.empty(n, dtype=object)
    for c in range(n_clusters):
        members = np.nonzero(cluster_id == c)[0]
        counts = {t: int(np.sum(per_building[members] == t))
                  for t in TREATMENT_ORDER}
        majority = max(TREATMENT_ORDER, key=lambda t: counts[t])
        final[members] = majority
    return final, cluster_id


def capacity_assessment(
    current_far: float,
    planned_far: float,
    lo_ratio: float = DEMOLISH_RATIO,
    hi_ratio: float = RENOVATE_RATIO,
) -> Dict[str, Optional[Union[float, str]]]:
    """Method 3 — FAR-ratio based 留-改-拆.

    ``ratio = current_far / planned_far`` — the lower the current density
    relative to the plan, the higher the redevelopment value:

    - ratio < ``lo_ratio``            -> demolish (拆, major redevelopment)
    - lo_ratio <= ratio < ``hi_ratio`` -> renovate (改, partial upgrade)
    - ratio >= ``hi_ratio``           -> retain (留, density near plan)

    ``planned_far <= 0`` means there is no capacity target -> retain with
    ``ratio`` None and a note.

    Returns a dict with ``ratio``, ``treatment``, ``note``.
    """
    if planned_far <= 0:
        return {"ratio": None, "treatment": "retain",
                "note": "no planned capacity target"}
    ratio = current_far / planned_far
    if ratio < lo_ratio:
        treatment = "demolish"
    elif ratio < hi_ratio:
        treatment = "renovate"
    else:
        treatment = "retain"
    return {"ratio": ratio, "treatment": treatment, "note": "ok"}
