from pathlib import Path

import cartopy  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.colors import LogNorm, Normalize  # type: ignore
from pyresample.geometry import AreaDefinition  # type: ignore


NORM = {
    "bias": Normalize(vmin=-1, vmax=1),
    "fse": Normalize(vmin=0, vmax=2),
    "pod": Normalize(vmin=0, vmax=1),
    "pofd": Normalize(vmin=0, vmax=1),
    "corr": Normalize(vmin=-1, vmax=1),
    "rr": LogNorm(vmin=0.01, vmax=10),
}


def plot_stats(
    product: str,
    metrics: dict[str, np.ndarray],
    area: AreaDefinition,
    stats_dir: Path,
) -> None:
    """Plot statistics."""

    crs = area.to_cartopy_crs()

    for metric, values in metrics.items():
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=crs)
        ax.gridlines(draw_labels=False, x_inline=False, y_inline=False)
        c = ax.imshow(
            values[1],
            transform=crs,
            extent=crs.bounds,
            cmap="viridis",
            norm=NORM[metric]
        )
        ax.add_feature(cartopy.feature.COASTLINE)
        gl = ax.gridlines(
            crs=cartopy.crs.PlateCarree(),
            draw_labels=True,
            x_inline=False,
            y_inline=False
        )
        gl.right_labels = False
        gl.top_labels = False
        gl.left_labels = True
        gl.bottom_labels = True
        fig.colorbar(c)
        fig.tight_layout()
        outfile = stats_dir / f"{product}_{metric}.png"
        plt.savefig(outfile)
        plt.close()


def plot_scene(
    rr_prx: np.ndarray,
    rr_baltrad: np.ndarray,
    area: AreaDefinition,
    outfile: Path,
) -> None:
    """Plot the scene."""

    crs = area.to_cartopy_crs()
    fig = plt.figure(figsize=(12, 6))
    for idx, rr in enumerate([rr_baltrad, rr_prx]):
        ax = fig.add_subplot(1, 2, idx + 1, projection=crs)
        ax.gridlines(draw_labels=False, x_inline=False, y_inline=False)
        c = ax.imshow(
            np.clip(rr, min=NORM["rr"].vmin),
            transform=crs,
            extent=crs.bounds,
            norm=NORM["rr"],
            cmap="PuBu",
        )
        ax.imshow(  # make low quality data to appear in gray
            np.isfinite(rr),
            transform=crs,
            extent=crs.bounds,
            vmin=0,
            vmax=1,
            cmap="gray",
            alpha=0.1,
        )
        ax.add_feature(cartopy.feature.COASTLINE)
        gl = ax.gridlines(
            crs=cartopy.crs.PlateCarree(),
            draw_labels=True,
            x_inline=False,
            y_inline=False
        )
        gl.right_labels = False
        gl.top_labels = False
        gl.left_labels = True
        gl.bottom_labels = True
        fig.colorbar(c, shrink=0.8)

    fig.tight_layout()
    plt.savefig(outfile)
