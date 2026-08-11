"""
Reusable plotting utilities for professional urban planning figures.

Provides:
- ``plot_land_use`` — professional land-use plan figure
- ``add_scale_bar`` — scale bar using matplotlib-scalebar
- ``add_north_arrow`` — simple north arrow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from matplotlib_scalebar.scalebar import ScaleBar
from shapely.geometry import Polygon

try:
    from .generation import LAND_USE_COLORS, LAND_USE_LABELS  # noqa: F401
except ImportError:
    from generation import LAND_USE_COLORS, LAND_USE_LABELS  # noqa: F401


def _setup_chinese_font():
    """Attempt to configure a Chinese-capable font."""
    import matplotlib

    chinese_fonts = [
        "Heiti SC",
        "STHeiti",
        "Songti SC",
        "PingFang SC",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
    ]
    for font in chinese_fonts:
        try:
            matplotlib.font_manager.findfont(font, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return font
        except Exception:
            continue

    # Fallback
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"


def add_scale_bar(ax, transform, length_m: float = 500):
    """
    Add a scale bar to a matplotlib Axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes.
    transform : matplotlib transform
        CRS transform for the axes (usually from the GeoDataFrame plot).
    length_m : float
        Length of the scale bar in metres.
    """
    scalebar = ScaleBar(
        1,  # 1 metre per unit (data is in metres)
        "m",
        length_fraction=None,
        scale_loc="bottom",
        location="lower left",
        fixed_value=length_m,
        fixed_units="m",
        box_alpha=0,
        color="black",
        font_properties={"size": 9},
    )
    ax.add_artist(scalebar)


def add_north_arrow(ax, x: float = 0.92, y: float = 0.92, size: float = 0.04):
    """
    Add a simple north arrow to a matplotlib Axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes.
    x, y : float
        Axes-relative coordinates (0-1).
    size : float
        Arrow size in axes-relative units.
    """
    ax.annotate(
        "N",
        xy=(x, y - size),
        xytext=(x, y),
        arrowprops=dict(
            arrowstyle="->", lw=2.0, color="black"
        ),
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
        xycoords="axes fraction",
    )


def plot_land_use(
    gdf: gpd.GeoDataFrame,
    boundary: Polygon,
    title: str,
    output_path: str,
    land_use_colors: dict = None,
    land_use_labels: dict = None,
    dpi: int = 300,
    color_col: str = "land_use_code",
):
    """
    Render a professional land-use plan figure.

    Parameters
    ----------
    gdf : GeoDataFrame
        Land parcels with a ``land_use_code`` column.
    boundary : Polygon
        The overall boundary polygon.
    title : str
        Figure title (Chinese supported if font available).
    output_path : str
        File path for the output PNG.
    land_use_colors : dict, optional
        land_use_code → hex colour.
    land_use_labels : dict, optional
        land_use_code → display label.
    dpi : int
        Output resolution.
    color_col : str
        Column in *gdf* that maps to colour keys.
    """
    _setup_chinese_font()

    if land_use_colors is None:
        land_use_colors = LAND_USE_COLORS
    if land_use_labels is None:
        land_use_labels = LAND_USE_LABELS

    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot each land-use code separately and build legend handles
    codes_present = sorted(gdf[color_col].unique())
    legend_handles = []
    for code in codes_present:
        subset = gdf[gdf[color_col] == code]
        color = land_use_colors.get(code, "#CCCCCC")
        label = land_use_labels.get(code, code)
        subset.plot(ax=ax, color=color, edgecolor="#333333", linewidth=0.3)
        legend_handles.append(
            mpatches.Patch(color=color, label=label)
        )

    # Boundary outline
    gpd.GeoSeries([boundary]).plot(
        ax=ax, facecolor="none", edgecolor="#222222", linewidth=1.5
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    # Legend with explicit handles
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
        ncol=2,
        title="用地分类",
        title_fontsize=9,
    )

    # Scale bar and north arrow
    add_scale_bar(ax, ax.transData, length_m=500)
    add_north_arrow(ax)

    # Source note and watermark
    fig.text(
        0.01,
        0.01,
        u"来源: urban-spatial-tooling 探索项目 | 模拟数据, 非官方规划",
        fontsize=7,
        color="gray",
        ha="left",
    )
    fig.text(
        0.5,
        0.5,
        u"临时模拟",
        fontsize=48,
        color="gray",
        alpha=0.15,
        ha="center",
        va="center",
        rotation=30,
        transform=fig.transFigure,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path
