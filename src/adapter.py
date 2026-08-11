"""
Adapter layer: haidian-format GeoJSON features -> pipeline check inputs.

Haidian submissions carry planning attributes (``land_use_code``,
``building_type``, ``floors_estimated``, ``area_sqm_declared``) but NOT
the technical-review pipeline's computed attributes (``height_m``,
``spacing_to_south_m``, ``min_setback_m``, ``adjacent_road_class``).
This module bridges the two formats:

- :func:`adapt_building` estimates building height from storeys and
  measures south spacing / road setback from the geometry itself
  (projected to EPSG:4548 metres);
- :func:`adapt_land_use` maps haidian ``land_use_code`` to the
  pipeline's runoff-oriented ``land_use_type`` (used by
  ``check_sponge_city``);
- :func:`load_hadian_submission` loads a submission's ``geometry/``
  directory into per-layer feature lists.

Every estimate is traceable: adapted properties carry ``<field>_source``
keys and free-text ``_adapter_notes`` so a reviewer can distinguish
measured values from estimates.  When a value cannot be measured at all
it is left absent (never fabricated) — the pipeline then fails closed
where that is the correct behaviour (solar spacing) or applies its own
default (setback, see ``check_setback``).

Data conventions referenced here:
- haidian ``brief/site-package/enums/land_use_codes.json`` (07 居住,
  08 公共服务, 09 商业, 10 工矿, 12 交通, 13 公用设施, 14 绿地, 16 留白)
- haidian ``brief/site-package/enums/road_classes.json``
- haidian ``brief/site-package/enums/building_types.json``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from shapely.geometry import shape as shapely_shape
from shapely.geometry.base import BaseGeometry

try:
    from .projection import CRS_4326, CRS_4548, transform_geometry
except ImportError:  # direct module import (src/ on sys.path)
    from projection import CRS_4326, CRS_4548, transform_geometry

# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------

# Reference floor heights (m) per haidian building_type.  Estimate only —
# values are common design defaults (residential 3.0 m, office 3.9 m,
# retail 4.5 m, cultural 5.4 m ...).
DEFAULT_FLOOR_HEIGHT_M = 3.6
FLOOR_HEIGHT_M_BY_TYPE: Dict[str, float] = {
    'residential': 3.0,
    'talent_apartment': 3.0,
    'existing_retained': 3.6,
    'mixed_use': 3.6,
    'community_service': 3.6,
    'office': 3.9,
    'ai_r_and_d': 3.9,
    'incubator': 3.9,
    'education': 3.9,
    'lab': 4.2,
    'retail': 4.5,
    'mobility_hub': 5.0,
    'cultural': 5.4,
}

# land_use_code prefix -> check_sponge_city land_use_type (runoff phi).
# Prefixes follow haidian's subset of the MNR land-use classification.
LAND_USE_PREFIX_TO_TYPE: Dict[str, str] = {
    '07': 'residential',  # 居住用地
    '08': 'public',       # 公共管理与公共服务用地
    '09': 'commercial',   # 商业服务业用地
    '10': 'industrial',   # 工矿用地
    '12': 'road',         # 交通运输用地 (1207 城镇村道路用地)
    '13': 'public',       # 公用设施用地
    '14': 'green',        # 绿地与开敞空间用地
    '16': 'green',        # 留白用地 — reserved, low-runoff open land
    '23': 'water',        # 水域
}

# haidian road_type -> pipeline check_setback road class.
# The pipeline table is {'arterial': 15, 'secondary': 10, 'local': 5};
# haidian enum values that have no pipeline row are folded onto the
# nearest class (expressway -> arterial, all slow-mobility classes ->
# local).  'major' is the legacy value used by the test submission.
ROAD_CLASS_PIPELINE: Dict[str, str] = {
    'expressway': 'arterial',
    'arterial': 'arterial',
    'major': 'arterial',
    'secondary': 'secondary',
    'branch': 'local',
    'minor': 'local',
    'local': 'local',
    'local_access': 'local',
    'pedestrian': 'local',
    'cycleway': 'local',
    'greenway': 'local',
    'transit_connection': 'local',
}

# geometry/ filename stem -> canonical layer key in the loader result.
LAYER_FILE_ALIASES: Dict[str, str] = {
    'site_boundary': 'site_boundary',
    'buildings': 'buildings',
    'building_footprint': 'buildings',
    'land_use': 'land_use',
    'roads': 'roads',
    'road_centerline': 'roads',
    'green_space': 'green_space',
    'public_space': 'public_space',
    'key_areas': 'key_areas',
    'constraints': 'constraints',
    'phasing': 'phasing',
    'water_system': 'water_system',
    'water': 'water_system',
}

DEFAULT_SITE_AREA_SQM = 11_400_000.0  # haidian declared site area fallback


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------


def _feature_id(feature: Dict[str, Any]) -> str:
    """Best-effort feature identifier from properties.id / feature id."""
    props = feature.get('properties') or {}
    return str(props.get('id') or feature.get('id') or '')


def _geometry_of(feature: Dict[str, Any]) -> Optional[BaseGeometry]:
    """Shapely geometry of a GeoJSON feature (raw, unprojected)."""
    geom = feature.get('geometry')
    if geom is None:
        return None
    try:
        return shapely_shape(geom)
    except Exception:
        return None


def _projected(geom: Optional[BaseGeometry]) -> Optional[BaseGeometry]:
    """Project a WGS84 (EPSG:4326) geometry to EPSG:4548 metres.

    Returns None when projection fails so callers can degrade gracefully
    instead of crashing the whole adaptation.
    """
    if geom is None:
        return None
    try:
        return transform_geometry(geom, CRS_4326, CRS_4548)
    except Exception:
        return None


def _south_spacing_m(building_id: str,
                     bldg_geom: BaseGeometry,
                     neighbors: Sequence[Dict[str, Any]],
                     site_boundary_geom: Optional[BaseGeometry]) -> tuple:
    """Measure the clear north-south gap to the nearest south building.

    The candidate set is every neighbor whose centroid lies south of the
    building's centroid; the spacing is the smallest positive gap between
    this building's south edge and the neighbor's north edge (EPSG:4548
    metres).  A neighbor whose N-S extent overlaps the building clamps
    the gap to 0.0 (conservative: cannot be shown compliant).  When no
    south neighbor exists, the fallback is the gap to the site boundary's
    south edge — no in-submission obstruction, so the constraint is
    bounded only by the site edge.

    Returns
    -------
    tuple
        ``(spacing_m, source)`` where *source* is one of
        ``'geometry_nearest_south'``, ``'geometry_nearest_south_overlapping'``,
        ``'geometry_site_boundary_fallback'``, or ``(None, None)`` when
        nothing can be measured.
    """
    south_geoms: List[BaseGeometry] = []
    for n in neighbors:
        if _feature_id(n) == building_id:
            continue
        g = _projected(_geometry_of(n))
        if g is None or g.is_empty:
            continue
        if g.centroid.y >= bldg_geom.centroid.y:
            continue  # not to the south
        south_geoms.append(g)

    if south_geoms:
        my_miny = bldg_geom.bounds[1]
        gaps = [my_miny - g.bounds[3] for g in south_geoms]
        min_gap = min(gaps)
        if min_gap < 0.0:
            return 0.0, 'geometry_nearest_south_overlapping'
        return float(min_gap), 'geometry_nearest_south'

    if site_boundary_geom is not None:
        gap = bldg_geom.bounds[1] - site_boundary_geom.bounds[1]
        return float(max(gap, 0.0)), 'geometry_site_boundary_fallback'

    return None, None


def _nearest_road(geom: BaseGeometry,
                  roads: Sequence[Dict[str, Any]]) -> tuple:
    """Distance (m) to the nearest road centerline and its adapted class.

    Returns ``(distance_m, pipeline_road_class)`` or ``(None, None)``
    when no road geometry can be measured.
    """
    best_dist: Optional[float] = None
    best_class: Optional[str] = None
    for r in roads:
        g = _projected(_geometry_of(r))
        if g is None or g.is_empty:
            continue
        d = float(geom.distance(g))
        if best_dist is None or d < best_dist:
            best_dist = d
            raw_class = str((r.get('properties') or {}).get('road_type', 'local'))
            best_class = ROAD_CLASS_PIPELINE.get(raw_class, 'local')
    return best_dist, best_class


def _estimate_height_m(props: Dict[str, Any]) -> Optional[float]:
    """Height from ``height_m`` or storeys * reference floor height."""
    declared = props.get('height_m')
    if isinstance(declared, (int, float)) and declared > 0:
        return float(declared)
    storeys = (props.get('floors_estimated')
               or props.get('storeys')
               or props.get('floor_count'))
    if isinstance(storeys, (int, float)) and storeys > 0:
        btype = str(props.get('building_type', ''))
        per_floor = FLOOR_HEIGHT_M_BY_TYPE.get(btype, DEFAULT_FLOOR_HEIGHT_M)
        return float(storeys) * per_floor
    return None


# ---------------------------------------------------------------------------
# Feature adapters
# ---------------------------------------------------------------------------


def adapt_building(feature: Dict[str, Any],
                   neighbors: Optional[Sequence[Dict[str, Any]]] = None,
                   roads: Optional[Sequence[Dict[str, Any]]] = None,
                   site_boundary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Adapt a haidian BUILDING_FOOTPRINT feature for the solar/setback checks.

    Haidian buildings carry ``building_type`` and ``floors_estimated``
    but never ``height_m`` / ``spacing_to_south_m`` / ``min_setback_m``
    / ``adjacent_road_class``.  This adapter fills the gaps:

    - ``height_m`` — declared value, else ``floors_estimated`` times a
      reference floor height for the building type;
    - ``spacing_to_south_m`` — measured from geometry when *neighbors*
      (other raw building features) are supplied: clear N-S gap to the
      nearest building to the south; falls back to the gap to the site
      boundary's south edge when the building has no south neighbor;
    - ``min_setback_m`` / ``adjacent_road_class`` — measured to the
      nearest road centerline when *roads* (raw road features) are
      supplied; the road type is normalized to the pipeline's class set.

    Values that cannot be measured are left absent rather than
    fabricated; ``_adapter_notes`` and the ``<field>_source`` keys record
    how each value was obtained.

    Parameters
    ----------
    feature : dict
        Raw haidian GeoJSON feature.
    neighbors : list of dict, optional
        Other raw building features used to measure solar spacing.
    roads : list of dict, optional
        Raw road features used to measure setback.
    site_boundary : dict, optional
        Raw site boundary feature used as south-edge fallback.

    Returns
    -------
    dict
        GeoJSON-style feature whose ``properties`` are the check inputs.
    """
    props = dict(feature.get('properties') or {})
    notes: List[str] = []

    # -- building type: talent apartments are housing for the solar check
    btype_original = str(props.get('building_type', ''))
    btype = ('residential' if btype_original in ('residential', 'talent_apartment')
             else btype_original)
    if btype_original == 'talent_apartment':
        notes.append('building_type talent_apartment -> residential (housing needs solar access)')

    # -- height
    declared_height = props.get('height_m')
    height = _estimate_height_m(props)
    if isinstance(declared_height, (int, float)) and declared_height > 0:
        height_source = 'declared'
    elif height is not None:
        height_source = 'estimated_from_storeys'
        storeys = props.get('floors_estimated') or props.get('storeys')
        notes.append(f'height_m estimated: {storeys} storeys * '
                     f'{FLOOR_HEIGHT_M_BY_TYPE.get(btype_original, DEFAULT_FLOOR_HEIGHT_M)} m/floor')
    else:
        height_source = None

    # -- solar spacing from geometry
    geom = _projected(_geometry_of(feature))
    spacing = props.get('spacing_to_south_m')
    spacing_source: Optional[str]
    if isinstance(spacing, (int, float)):
        spacing_source = 'declared'
    elif geom is not None:
        site_geom = _projected(_geometry_of(site_boundary)) if site_boundary else None
        spacing, spacing_source = _south_spacing_m(
            _feature_id(feature), geom, list(neighbors or []), site_geom)
        if spacing is not None:
            notes.append(f'spacing_to_south_m measured from geometry ({spacing_source})')
    else:
        spacing_source = None

    # -- setback to nearest road
    declared_setback = props.get('min_setback_m')
    declared_road_class = props.get('adjacent_road_class')
    min_setback: Optional[float] = None
    road_class: Optional[str] = None
    if isinstance(declared_setback, (int, float)):
        min_setback = float(declared_setback)
        road_class = ROAD_CLASS_PIPELINE.get(
            str(declared_road_class or 'local'), 'local')
    elif geom is not None and roads:
        min_setback, road_class = _nearest_road(geom, roads)
        if min_setback is not None:
            notes.append('min_setback_m measured to nearest road centerline')

    out_props: Dict[str, Any] = {
        'id': _feature_id(feature),
        'building_type': btype,
        'building_type_original': btype_original,
        'height_m': height,
        'height_source': height_source,
        'spacing_to_south_m': (round(spacing, 2) if spacing is not None else None),
        'spacing_source': spacing_source,
        'floors_estimated': props.get('floors_estimated'),
        'area_sqm': (round(geom.area, 2) if geom is not None else None),
        '_adapter_notes': notes,
    }
    if min_setback is not None:
        out_props['min_setback_m'] = round(min_setback, 2)
        out_props['adjacent_road_class'] = road_class

    return {
        'type': 'Feature',
        'id': _feature_id(feature),
        'properties': out_props,
        'geometry': feature.get('geometry'),  # original WGS84, traceability
    }


def adapt_land_use(feature: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt a haidian LAND_USE feature for the sponge-city check.

    Maps ``land_use_code`` (MNR classification, e.g. ``0701``, ``1401``)
    to the pipeline's runoff-oriented ``land_use_type`` used by
    ``check_sponge_city`` (residential / public / commercial /
    industrial / green / road / water), and produces ``area_sqm`` —
    declared area when present, else the EPSG:4548 projected area.

    Parameters
    ----------
    feature : dict
        Raw haidian GeoJSON feature with ``properties.land_use_code``.

    Returns
    -------
    dict
        GeoJSON-style feature with adapted ``properties``.
    """
    props = dict(feature.get('properties') or {})
    code = str(props.get('land_use_code', ''))

    lu_type: Optional[str] = None
    for prefix, mapped in sorted(LAND_USE_PREFIX_TO_TYPE.items(),
                                 key=lambda kv: len(kv[0]), reverse=True):
        if code.startswith(prefix):
            lu_type = mapped
            break
    notes: List[str] = []
    if lu_type is None:
        lu_type = 'residential'  # pipeline default phi (0.60)
        notes.append(f'land_use_code {code!r} unmapped -> pipeline default residential')

    area = props.get('area_sqm') or props.get('area_sqm_declared')
    if isinstance(area, (int, float)):
        area_source = 'declared'
    else:
        geom = _projected(_geometry_of(feature))
        area = geom.area if geom is not None else None
        area_source = 'geometry_projected' if area is not None else None
        if area is not None:
            notes.append('area_sqm computed from EPSG:4548 projected geometry')

    return {
        'type': 'Feature',
        'id': _feature_id(feature),
        'properties': {
            'id': _feature_id(feature),
            'layer': props.get('layer'),
            'land_use_code': code,
            'land_use_type': lu_type,
            'name_zh': props.get('name_zh'),
            'area_sqm': round(float(area), 2) if area is not None else None,
            'area_source': area_source,
            '_adapter_notes': notes,
        },
        'geometry': feature.get('geometry'),
    }


# ---------------------------------------------------------------------------
# Submission loader
# ---------------------------------------------------------------------------


def _resolve_geometry_dir(submission_path: str) -> Path:
    """Resolve *submission_path* to the directory holding the .geojson files.

    Accepts either a submission root containing a ``geometry/``
    subdirectory (haidian layout) or the ``geometry`` directory itself.

    Raises
    ------
    FileNotFoundError
        When neither exists or the directory holds no .geojson files.
    """
    path = Path(submission_path)
    if (path / 'geometry').is_dir():
        return path / 'geometry'
    if path.is_dir() and list(path.glob('*.geojson')):
        return path
    raise FileNotFoundError(
        f'no geometry directory found at {submission_path!r} '
        f'(looked for {path / "geometry"} and {path})')


def load_hadian_submission(submission_path: str) -> Dict[str, Any]:
    """Load a haidian submission's ``geometry/`` directory into layers.

    Every ``*.geojson`` file is read and its features bucketed by a
    canonical layer key derived from the file name (see
    :data:`LAYER_FILE_ALIASES`).  The returned dict always contains the
    four pipeline keys:

    - ``buildings`` — list of BUILDING_FOOTPRINT features
    - ``land_use`` — list of LAND_USE features
    - ``site_boundary`` — single feature dict (or None) — the reference
      boundary: the first SITE_BOUNDARY-layer feature, else the first
      feature of ``site_boundary.geojson``
    - ``roads`` — list of road features

    Additional layers (``green_space``, ``public_space``, ...) are
    included under their canonical keys when present.

    Parameters
    ----------
    submission_path : str
        Submission root directory (containing ``geometry/``) or the
        geometry directory itself.

    Returns
    -------
    dict
        Layer buckets keyed by canonical layer name.
    """
    geo_dir = _resolve_geometry_dir(submission_path)
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(geo_dir.glob('*.geojson')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f'cannot read {path}: {exc}') from exc
        features = data.get('features', []) if isinstance(data, dict) else []
        key = LAYER_FILE_ALIASES.get(path.stem, path.stem)
        buckets.setdefault(key, []).extend(features)

    # -- site boundary: SITE_BOUNDARY layer preferred, else first feature
    site_features = buckets.get('site_boundary', [])
    site_boundary: Optional[Dict[str, Any]] = None
    for f in site_features:
        if (f.get('properties') or {}).get('layer') == 'SITE_BOUNDARY':
            site_boundary = f
            break
    if site_boundary is None and site_features:
        site_boundary = site_features[0]

    result: Dict[str, Any] = {
        'buildings': buckets.get('buildings', []),
        'land_use': buckets.get('land_use', []),
        'site_boundary': site_boundary,
        'roads': buckets.get('roads', []),
    }
    for key, features in buckets.items():
        if key not in result:
            result[key] = features
    return result


__all__ = [
    "adapt_building", "adapt_land_use", "load_hadian_submission",
    "DEFAULT_FLOOR_HEIGHT_M", "FLOOR_HEIGHT_M_BY_TYPE",
    "LAND_USE_PREFIX_TO_TYPE", "ROAD_CLASS_PIPELINE",
]
