# experimental/ — Archived GPU (Metal/MPS) Acceleration Experiments

These modules were **Apple Silicon GPU (Metal/MPS) acceleration experiments**
for urban spatial operations, benchmarked in `notebooks/08-*` and
`notebooks/09-*`.

**STATUS: ARCHIVED — NOT used in any production path.** All production code
in this repository is CPU-only (`shapely`/GEOS, `pyproj`, `numpy`). Do not
import these modules from production code, and do not re-introduce them
without re-running the benchmarks.

## Modules

| Module | Notebook | What it tested |
|---|---|---|
| `metal_projection.py` | `notebooks/08-metal-projection-benchmark.ipynb` | Batch EPSG:4326 → EPSG:4548 projection via a local affine/quadratic polynomial fit applied as one batched MPS matmul |
| `metal_spatial.py` | `notebooks/09-point-in-polygon-benchmark.ipynb` | Batch point-in-polygon (even-odd ray casting) on MPS, vs shapely/GEOS |

## Conclusion: GPU is SLOWER than CPU for most urban spatial operations

### 08 — Projection (mixed; only large workloads win)

- MPS batch projection beats CPU only on very large workloads: crossover at
  ~10k–100k points; ≥100k points gives 3–4.5x end-to-end (up to ~30x
  GPU-only), but numpy assembly + transfer dominates the end-to-end time.
- Precision: quadratic polynomial < 1 mm vs pyproj; linear affine ≈ 1.65 m —
  fails the 1 m absolute-coordinate tolerance, so the quadratic model
  (same single matmul) is mandatory.
- Shoelace area on GPU is bandwidth-bound — no standalone advantage; it is
  only "free" as part of a combined projection + area pipeline.

### 09 — Point-in-Polygon (GPU loses in practice)

- Correctness: MPS even-odd ray casting matches GEOS exactly (0 mismatch on
  main polygon, holes, 50-parcel coverage, 500-parcel attribution).
- BUT speedup is data-distribution dependent: GEOS's bbox fast path handles
  "most points outside the polygon" global scans at GPU parity (0.8–1.1x);
  MPS only reaches 1.4–1.6x when points are dense inside the bbox, and falls
  to 0.4–0.9x as edge counts grow (GEOS edge-level STRtree keeps O(N·log E)).
- Attribution (N points × P parcels): shapely STRtree wins by 72.6x — the
  per-parcel GPU loop is O(N·P), a structural disadvantage that cannot be
  fixed on the GPU.
- GPU's 20x+ advantage over a naive numpy implementation only proves the
  right fix is to switch to GEOS, not to the GPU.

**Verdict:** projection is a dense-matmul GPU sweet spot, but
point-in-polygon is a sparse-comparison problem where GEOS is the sweet spot.
The probes deliberately drew this line with data: not every operator deserves
a GPU. Recommended hybrid (never built): GPU for projection only, everything
else on GEOS.

## Upstream bug discovered (torch 2.12.0 MPS GEMM)

torch 2.12.0 and 2.13.0-dev on Apple silicon (observed on M5 Max)
**intermittently mis-compute the batched matmul `A @ B`**, reading the right
operand as if it were stored transposed (column-major). The failure is
deterministic within a process but flips between processes / over time
(GPU-wide state, not a kernel-local issue).

Workaround implemented in `metal_projection.py`: probe the GPU once at
process start (compare a small matmul against CPU), apply a `.t()`
compensation on the right operand when needed, and re-verify on the first
real batch. `metal_spatial.py` only uses elementwise ops + reductions, so it
is inherently unaffected — but any future GEMM-based GPU code on this
machine MUST keep the probe + compensation + verify pattern.

## Re-running the archived notebooks

`notebooks/08-*` and `notebooks/09-*` import these modules as
`from src.metal_projection import ...` / `from src.metal_spatial import ...`.
After archiving, re-running them requires updating the import paths to
`src.experimental.metal_projection` / `src.experimental.metal_spatial`.
