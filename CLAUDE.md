# Urban Spatial Tooling Exploration

私人技术探索项目——城市设计领域所需的几何生成、空间分析和专业可视化能力。

## 背景

来自 [haidian](../haidian) 项目的技术缺口分析。当前系统缺少三大核心能力：

1. **空间数据生成** — 不能生成有意义的城市设计 GeoJSON（只有矩形占位符）
2. **空间分析管线** — 拓扑检查、面积计算、指标复算未实现
3. **专业可视化** — 只能生成简单矩形图，无法输出专业规划图纸

## 探索目标

### 1. 城市空间数据生成 (GeoJSON Generation)

从"概念描述"到"空间几何"的生成能力。

- **用地布局生成**:
  - 给定用地功能需求（比例、总面积），在边界内生成无间隙无重叠的用地分区
  - 算法: 基于约束的矩形分区、Voronoi 微调、道路网络引导分割
  - 输出: 符合 `land_use_code` 枚举的 GeoJSON Polygon 集合
  - 保证: 完全覆盖、无重叠、面积在容差内

- **建筑基底生成**:
  - 在用地分区内生成建筑 footprint
  - 考虑建筑类型、朝向、间距（日照间距）
  - 输出: 带 `building_type` 属性的 GeoJSON Polygon

- **道路网络生成**:
  - 基于现有道路骨架，生成规划道路网络
  - 道路等级层次（主干路/次干路/支路 + 慢行专用道）
  - 输出: 带 `road_class` 的 GeoJSON LineString

- **绿地与公共空间生成**:
  - 绿地系统结构（廊道 + 斑块 + 节点）
  - 公共空间序列和等级
  - 输出: 带分类属性的 GeoJSON Polygon

关键库:
- `shapely` — 几何对象操作 (Polygon, Point, LineString)
- `geopandas` — GeoDataFrame 管理和空间操作
- `scipy.spatial.Voronoi` — 空间分割
- `networkx` — 道路图分析

### 2. 空间分析管线 (Spatial Analysis)

约束引擎的核心计算能力。

- **坐标转换**:
  - EPSG:4326 (WGS84 lat/lng) ↔ EPSG:4548 (CGCS2000 Gauss-Kruger CM 117E)
  - `pyproj.Transformer` — 精确投影变换
  - 验证: 转换后面积与官方声明值比较

- **拓扑检查**:
  - 覆盖检查: 一组 Polygon 是否完全覆盖边界（无间隙）
  - 重叠检查: 任意两个 Polygon 是否重叠
  - 包含检查: 所有 feature 是否在边界内
  - 实现: `shapely.ops.unary_union` + `.covers()` + `.intersects()`

- **面积计算**:
  - EPSG:4548 下的精确面积（sqm → ha → km²）
  - 面积容差验证（±5% site, ±10% 关键区域）
  - 批量面积统计（按用地类型、按片区）

- **指标计算**:
  - 容积率 (FAR) = 总建筑面积 / 用地面积
  - 建筑密度 = 建筑基底面积 / 用地面积
  - 绿地率 = 绿地面积 / 用地面积
  - 道路网密度 = 道路总长度 / 用地面积
  - 每个指标: 从 GeoJSON → 计算结果 → 可复算验证

关键库:
- `pyproj` — CRS 转换
- `shapely` — 几何运算
- `geopandas` — 批量空间操作
- `numpy` — 数值计算

### 3. 专业规划图纸生成 (Figure Generation)

从"数据"到"图纸"的可视化能力。

- **图纸类型** (haidian 需要的 5 张):
  1. **总体概念图** (Site Overview) — 区位、范围、周边关系
  2. **空间结构图** (Land Use Structure) — 用地图、功能分区、结构轴线
  3. **重点区索引图** (Key Areas) — 三区两翼放大、标注
  4. **交通蓝绿系统图** (Mobility + Blue-Green) — 道路网络 + 绿地水系叠加
  5. **指标证据链图** (Metrics Evidence) — 指标可视化 + 计算来源标注

- **图纸专业标准**:
  - 标题 (title) + 图例 (legend) + 比例尺 (scale bar) + 指北针 (north arrow)
  - 来源标注 (source note) + provisional 标记
  - 配色: 专业规划色系（非默认色板）
  - 输出: 300dpi PNG，符合 A3/A0 打印要求
  - 中文标注: 支持中文字体渲染

关键库:
- `matplotlib` — 基础绑图 + 自定义布局
- `geopandas.plot()` — GeoDataFrame 快速渲染
- `contextily` — 底图（OSM/卫星图）
- `matplotlib-scalebar` — 比例尺
- `Pillow` (PIL) — 图像后处理、合成

### 4. 从数据到文档的完整管线

探索端到端的数据流动：

```
GeoJSON 图层
  ├→ pyproj 投影 → shapely 面积/拓扑 → metrics.json
  ├→ geopandas + matplotlib → PNG 图纸
  └→ geopandas.explore() → interactive HTML map
```

每个步骤必须是可复算、可验证的。

## 项目结构（现状）

```
urban-spatial-tooling/
├── CLAUDE.md
├── notebooks/                   ← Jupyter notebooks for exploration
│   ├── 01-projection-and-area.ipynb            — 投影与面积
│   ├── 02-land-use-generation.ipynb            — 用地生成
│   ├── 03-building-generation.ipynb            — 建筑生成
│   ├── 04-road-network.ipynb                   — 道路网络
│   ├── 05-topology-checks.ipynb                — 拓扑验证
│   ├── 06-metrics-computation.ipynb            — 指标计算
│   ├── 07-professional-figures.ipynb           — 图纸渲染
│   ├── 08-metal-projection-benchmark.ipynb     — GPU 投影 benchmark（已归档）
│   ├── 09-point-in-polygon-benchmark.ipynb     — GPU 点面判定 benchmark（已归档）
│   ├── 10-solar-analysis.ipynb                 — 日照分析
│   ├── 11-view-corridor-analysis.ipynb         — 视廊分析
│   ├── 12-sponge-city.ipynb                    — 海绵城市
│   ├── 13-fifteen-minute-living-circle.ipynb   — 15 分钟生活圈
│   ├── 14-building-setback.ipynb               — 建筑退线
│   ├── 15-three-lines-compliance.ipynb         — 三区三线合规
│   └── 16-integrated-pipeline.ipynb            — 集成技术审查管线
├── src/                         ← Reusable code extracted from notebooks
│   ├── projection.py / generation.py / topology.py / visualization.py
│   ├── solar.py / compliance.py / sponge.py / living_circle.py / setback.py
│   ├── pipeline.py               — 技术审查 AND 门
│   └── experimental/             ← 已归档 GPU 实验（metal_projection, metal_spatial）
├── tests/                       ← pytest 测试（projection/topology/solar/compliance/pipeline）
├── data/                        ← Test data (use haidian provisional bounds as fixture)
│   └── haidian-boundary.geojson
└── outputs/                     ← Generated figures for review
```

## 技术栈

```
pyproj >= 3.6
shapely >= 2.0
geopandas >= 0.14
matplotlib >= 3.8
contextily >= 1.5
Pillow >= 10.0
numpy >= 1.26
jupyter
```

## 学习路径

1. **先跑通投影计算** — 读取 haidian 的 provisional_boundaries.geojson，在 EPSG:4548 下算面积，验证与官方值（11.4km²）的偏差
2. **再实现拓扑检查** — 用 shapely 实现覆盖/重叠/包含三类检查，用手工构造的测试数据验证
3. **尝试空间生成** — 从最简单的矩形分割开始，在 haidian 边界内生成模拟的用地分区
4. **实现指标计算** — 从生成的 GeoJSON 计算所有指标，输出 metrics.json
5. **学习专业图纸** — 用 matplotlib 渲染用地规划图，模仿专业配色和标注
6. **集成封装** — 将可复用的代码提取为 `src/` 模块

## 与 haidian 项目的关系

- 使用 haidian 的 `brief/site-package/geometry/` 数据作为测试 fixture
- 成熟后，将 `src/` 模块作为 haidian 项目的依赖或参考实现
- 填补 haidian 的 TECH-006 (GeoJSON 生成)、TECH-007 (指标计算)、TECH-008 (图纸生成) 缺口

## 学习路径进度

### [x] 1. 投影计算 — `notebooks/01-projection-and-area.ipynb`
- 读取 haidian provisional_boundaries.geojson
- EPSG:4326 -> EPSG:4548 投影变换
- 面积计算对比官方声明值 (11.4 km²)
- 偏差验证通过

### [x] 2. 拓扑检查 — `notebooks/05-topology-checks.ipynb`
- 覆盖检查 (coverage): `unary_union().covers(boundary)`
- 重叠检查 (overlaps): 逐对 intersection 检测
- 包含检查 (containment): `.within(boundary)`
- 用有意破损数据验证检测能力

### [x] 3. 空间生成 — `notebooks/02/03/04`
- **用地布局** (`02-land-use-generation.ipynb`): 网格细分 + 随机用地分配，无间隙无重叠
- **建筑基底** (`03-building-generation.ipynb`): 网格布点 + 裁剪到 parcel，按用地类型分配建筑类型
- **道路网络** (`04-road-network.ipynb`): 三级网格道路（次干道/支路/慢行道）

### [x] 4. 指标计算 — `notebooks/06-metrics-computation.ipynb`
- FAR (容积率): 2.1905
- 建筑密度: 22.47%
- 绿地率: 15.02%
- 道路网密度: 12.65 km/km²
- 输出 `outputs/metrics.json` — 每个指标可追溯到源几何数据

### [x] 5. 专业图纸 — `notebooks/07-professional-figures.ipynb`
- 用地规划图 (300dpi PNG)
- 用地+道路叠加图
- 指标证据链图 (4-panel dashboard)
- 标题/图例/比例尺/指北针/来源标注/水印

### [x] 6. 集成封装 — `src/` 模块
- `src/projection.py` — 投影变换、面积计算
- `src/topology.py` — 覆盖/重叠/包含检查
- `src/generation.py` — 用地分区、建筑生成
- `src/visualization.py` — 专业图纸渲染

### [x] 7. GPU 加速实验（负结果，已归档） — `notebooks/08/09`
- **08-metal-projection-benchmark.ipynb** — MPS 批量投影 benchmark
  - 二次多项式拟合 + 单次批量 matmul；二次多项式精度 < 1 mm（仿射 1.65 m 不达标）
  - 交叉点 10k–100k 点；≥100k 点端到端 3–4.5x（纯 GPU ~30x），端到端瓶颈在 numpy 装配 + 传输
  - 面积（shoelace）带宽受限，独立上 GPU 无优势
- **09-point-in-polygon-benchmark.ipynb** — MPS 批量点面判定 benchmark
  - 正确性 0 mismatch；但 GEOS bbox 快路径全域扫描 0.8–1.1x，仅点密集于 bbox 内 1.4–1.6x
  - 归属场景 shapely STRtree 72.6x 完胜（GPU 逐地块循环 O(N·P) 结构性劣势）
- 发现 torch 2.12.0/2.13.0-dev MPS GEMM 转置读取 bug（探测 + `.t()` 补偿 + 首调验证）
- **结论**：投影是 GPU 甜点，点面判定是 GEOS 甜点；生产路径 CPU-only，模块归档至 `src/experimental/`

### [x] 8. 专项空间分析 — `notebooks/10-15`
- **10-solar-analysis.ipynb** — 日照分析：GB 50180-2018 大寒日 ≥ 2h 标准、日照间距系数 D/H ≥ 1.6
- **11-view-corridor-analysis.ipynb** — 视廊分析：三山五园高度管控（数据门控）
- **12-sponge-city.ipynb** — 海绵城市：DB11/685-2021 年径流总量控制率 ≥ 85%（容积法，H = 33.6 mm）
- **13-fifteen-minute-living-circle.ipynb** — 15 分钟生活圈：TD/T 1062-2021 设施配置率
- **14-building-setback.ipynb** — 建筑退线：DB11/T 996-2013、GB 50016-2014 防火间距
- **15-three-lines-compliance.ipynb** — 三区三线合规检查

### [x] 9. 集成技术审查管线 — `notebooks/16-integrated-pipeline.ipynb`
- `src/pipeline.py`：CheckResult / Severity / Category / TechnicalReviewRunner
- AND 门聚合：CRITICAL/MAJOR 失败阻塞；NOT_ASSESSED（数据门控）不阻塞但如实上报
- A/B/C 分析分类框架（见下节）；输出 tech_review.json 供 haidian procedure.py 使用

### [x] 10. 模块提取与测试 — `src/` + `tests/`
- 从 notebooks 提取 10 个生产模块；GPU 实验 2 模块归档至 `src/experimental/`
- pytest 测试：projection / topology / solar / compliance / pipeline（含 fail-closed 与 AND 门逻辑覆盖）

## A/B/C 分析分类框架

对 10–15 号专项分析按计算特征分类（`src/pipeline.py` 的 `Category` 枚举），
决定每个检查的实现与验证策略：

| 类别 | 特征 | 检查 | 验证方式 |
|---|---|---|---|
| **A — COMPUTE_INTENSIVE (计算密集)** | 仿真驱动，确定性 CODE 可完整复算 | solar, sponge, living_circle | 参数 + 算法即可复算，结果可追溯 |
| **B — DATA_INTENSIVE (数据密集)** | 依赖官方 GIS 数据，缺数据无法验证 | view corridor, three-lines | 数据门控：缺官方数据时报 `NOT_ASSESSED`，绝不静默通过 |
| **C — PARAMETER_SENSITIVE (参数敏感)** | 结果由标准参数主导 | setback, runoff coeff | 参数显式暴露 + 引用标准条文，便于校准 |

用途：决定各检查在 AND 门中的失败行为——A/C 检查缺数据 fail-closed（硬失败），
B 检查缺数据 NOT_ASSESSED（不阻塞但如实上报），门绝不静默认证未运行的检查。

## GPU 加速实验结论（负结果：CPU 胜出）

notebooks 08/09 在 Apple M5 Max / Metal 4 上用 PyTorch MPS 验证 GPU 加速
城市空间计算，结论是**大部分算子不值得上 GPU，生产路径保持 CPU-only**
（相关模块已归档至 `src/experimental/`，详见其 README）：

- **投影（08）**：仅 ≥100k 点才有 3–4.5x 端到端收益（纯 GPU ~30x），交叉点
  在 10k–100k 点之间；端到端瓶颈是 numpy 装配 + 传输。二次多项式精度 < 1 mm，
  线性仿射 1.65 m 不满足 1 m 绝对坐标容差
- **点面判定（09）**：GEOS bbox 快路径在全域扫描场景 0.8–1.1x（GPU 无优势），
  仅点密集于多边形 bbox 内时 1.4–1.6x，边数增大后回落至 0.4–0.9x；
  归属（N 点 × P 地块）场景 shapely STRtree 72.6x 完胜
- **面积（08）**：带宽受限运算，独立上 GPU 无优势；仅作为"投影+面积"组合
  流水线（数据已在 GPU 上）的附带收益
- **上游 bug**：torch 2.12.0/2.13.0-dev MPS 间歇性将批量 matmul 右操作数按
  转置读取（进程内确定、进程间随机）。`metal_projection.py` 内置探测 +
  `.t()` 补偿 + 首调验证；本机任何新的 GEMM 代码必须保留该校验

### 产出物清单

```
data/haidian-boundary.geojson          — 测试边界数据
notebooks/01-16                        — 16 个探索 notebook（见学习路径进度）
src/projection.py                      — 投影变换、面积计算 (01)
src/generation.py                      — 用地分区、建筑生成 (02/03/04)
src/topology.py                        — 覆盖/重叠/包含检查 (05)
src/visualization.py                   — 专业图纸渲染 (07)
src/solar.py                           — 日照分析 (10)
src/compliance.py                      — 视廊 + 三区三线检查 (11/15)
src/sponge.py                          — 海绵城市容积法 (12)
src/living_circle.py                   — 15 分钟生活圈 (13)
src/setback.py                         — 建筑退线 (14)
src/pipeline.py                        — 技术审查 AND 门 (16)
src/experimental/                      — 已归档 GPU 实验 (08/09)
tests/                                 — pytest 测试（5 个模块）
outputs/*.png, metrics.json            — 生成的图纸和指标
```

### 下一步

- 成熟方法迁移到 haidian 项目：`src/pipeline.py` 的 TechnicalReviewRunner 对接
  haidian `procedure.py` 的 TechnicalAnalysis 阶段，输出 tech_review.json
- 实现基于道路网络引导的更真实用地分区 (Voronoi-based)
- 添加 contextily 底图支持（需要解决 gdal 依赖问题）
- GPU 路径维持归档状态；如需重启，先重跑 08/09 benchmark 并修复 notebook
  中 `src.metal_*` 的导入路径（现为 `src.experimental.metal_*`）

## Known Limitations

专家评审（expert review）发现的缺陷均已修复并保留记录，防止回归：

| 评审发现 | 状态 | 修复 |
|---|---|---|
| `check_solar` 对缺少 `spacing_to_south_m` 的建筑静默通过（缺数据 ≠ 合规） | 已修复 | fail-closed：缺数据记为硬失败（测试 `test_check_solar_missing_data_fails_closed`） |
| 数据门控检查（view corridor）缺官方 GIS 时按普通 FAIL 阻塞 AND 门 | 已修复 | 新增 `Severity.NOT_ASSESSED`：如实上报但不阻塞，门绝不静默认证未运行的检查 |
| `run_all()` 把全部 kwargs 盲目透传给每个检查 | 已修复 | `inspect.signature` 过滤：每个检查只接收自己声明的参数 |
| 检查抛异常会击穿 AND 门 | 已修复 | 异常记录为 CRITICAL error result，门永不崩溃 |
| torch 2.12.0 MPS GEMM 将右操作数按转置读取，批量投影结果随机错误 | 已规避 | 探测 + `.t()` 补偿 + 首调验证（`metal_projection.py`）；模块已归档 |
| 线性仿射投影近似误差 1.65 m > 1 m 绝对坐标容差 | 已修复 | 改用二次多项式（0.4 mm）；仿射仅限面积等相对量用途 |

残余限制（设计取舍，未修复）：
- 生产路径 CPU-only，无 GPU 加速（见 GPU 实验结论）
- contextily 底图依赖 gdal，未启用
- 部分检查使用标准参考值（setback 退距、sponge 径流系数）而非官方逐地块数据，
  进入正式审图流程前需用官方数据校准
- 面积/指标容差按探索期标准（±5% site / ±10% 关键区域），正式项目需按
  haidian 规范收紧
