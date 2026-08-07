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

## 项目结构建议

```
urban-spatial-tooling/
├── CLAUDE.md
├── notebooks/                   ← Jupyter notebooks for exploration
│   ├── 01-projection-and-area.ipynb
│   ├── 02-land-use-generation.ipynb
│   ├── 03-building-generation.ipynb
│   ├── 04-road-network.ipynb
│   ├── 05-topology-checks.ipynb
│   ├── 06-metrics-computation.ipynb
│   └── 07-professional-figures.ipynb
├── src/                         ← Reusable code extracted from notebooks
│   ├── projection.py
│   ├── topology.py
│   ├── generation.py
│   └── visualization.py
├── data/                        ← Test data (use haidian provisional bounds as fixture)
│   └── haidian-boundary.geojson
├── fixtures/                    ← Known-good test fixtures
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
- FAR (容积率): 2.07
- 建筑密度: 21.81%
- 绿地率: 18.06%
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

### 产出物清单

```
data/haidian-boundary.geojson          — 测试边界数据
notebooks/01-projection-and-area.ipynb — 投影与面积
notebooks/02-land-use-generation.ipynb — 用地生成
notebooks/03-building-generation.ipynb — 建筑生成
notebooks/04-road-network.ipynb        — 道路网络
notebooks/05-topology-checks.ipynb     — 拓扑验证
notebooks/06-metrics-computation.ipynb — 指标计算
notebooks/07-professional-figures.ipynb— 图纸渲染
src/projection.py, topology.py, generation.py, visualization.py — 可复用模块
outputs/*.png, metrics.json            — 生成的图纸和指标
```

### 下一步

- 成熟方法迁移到 haidian 项目作为正式依赖
- 实现基于道路网络引导的更真实用地分区 (Voronoi-based)
- 添加 contextily 底图支持（需要解决 gdal 依赖问题）
- 日照间距和退线约束的建筑生成
