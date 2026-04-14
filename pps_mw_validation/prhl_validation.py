from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import cartopy  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import xarray as xr  # type: ignore
from matplotlib.colors import LogNorm, Normalize  # type: ignore
from pyresample.geometry import AreaDefinition  # type: ignore


THRESHOLDS = [0, 0.01, 0.1, 1.0]  # precipitation rates in mm/h
NORM = {
    "bias": Normalize(vmin=-1, vmax=1),
    "fse": Normalize(vmin=0, vmax=2),
    "pod": Normalize(vmin=0, vmax=1),
    "pofd": Normalize(vmin=0, vmax=1),
    "corr": Normalize(vmin=-1, vmax=1),
    "rr": LogNorm(vmin=0.01, vmax=10),
}


@dataclass
class Stats:
    """Class for holding comparison stats."""

    thresholds: list[float]
    count_rain: np.ndarray
    count_clear: np.ndarray
    sum_baltrad: np.ndarray
    sum_prx: np.ndarray
    sum_square_baltrad: np.ndarray
    sum_square_prx: np.ndarray
    sum_product: np.ndarray
    sum_diff_square: np.ndarray
    true_detection: np.ndarray
    false_detection: np.ndarray
    product_tag: str

    @property
    def prx_mean(self) -> np.ndarray:
        """Get PRX mean precipitation rate."""
        return self.sum_prx / self.count_rain

    @property
    def prx_mean_square(self) -> np.ndarray:
        """Get PRX mean square precipitation rate."""
        return self.sum_square_prx / self.count_rain

    @property
    def baltrad_mean(self) -> np.ndarray:
        """Get BALTRAD mean precipitation rate."""
        return self.sum_baltrad / self.count_rain

    @property
    def baltrad_mean_square(self) -> np.ndarray:
        """Get BALTRAD mean square precipitation rate."""
        return self.sum_square_baltrad / self.count_rain

    @property
    def product_mean(self) -> np.ndarray:
        """Get BALTRAD and PRX product mean values."""
        return self.sum_product / self.count_rain

    @property
    def rmse(self) -> np.ndarray:
        """Get RMSE values."""
        return np.sqrt(self.sum_diff_square / self.count_rain)

    @property
    def correlation(self) -> np.ndarray:
        """Get correlation values."""
        prx_std = np.sqrt(self.prx_mean_square - self.prx_mean ** 2)
        baltrad_std = np.sqrt(self.baltrad_mean_square - self.baltrad_mean ** 2)
        covariance = self.product_mean - self.prx_mean * self.baltrad_mean
        return covariance / (prx_std * baltrad_std)

    @property
    def metrics(self) -> dict[str, np.ndarray]:
        """Get the main metrics."""
        return {
            "bias": self.prx_mean - self.baltrad_mean,
            "fse": self.rmse / self.baltrad_mean,
            "pod": self.true_detection / self.count_rain,
            "pofd": self.false_detection / self.count_clear,
            "corr":  self.correlation,
        }

    @property
    def validation_score(self) -> dict[str, dict[float, tuple[float, float]]]:
        """Get the overall validation score in a json friendly format."""
        return {
            metric: {
                t: stats for t, stats in zip(self.thresholds, self.get_stats(metric))
            } for metric in self.metrics
        }

    def get_stats(self, metric: "str") -> list[tuple[float, float]]:
        """Get mean and std of given metric."""
        values = np.where(np.isfinite(self.metrics[metric]), self.metrics[metric], np.nan)
        mean = np.nanmean(values, axis=(1, 2))
        std = np.nanstd(values, axis=(1, 2))
        return np.stack((mean, std), axis=1).tolist()

    @staticmethod
    def to_json(validation_score: dict[str, Any], outdir: Path) -> Path:
        """Write validation score to JSON file."""
        outfile = outdir / "baltrad_comparison_summary.json"
        with open(outfile, "w") as f:
            f.write(json.dumps(validation_score, indent=4))
        return outfile

    @classmethod
    def zeros(
        cls,
        image_shape: tuple[int, int],
    ) -> "Stats":
        """Get stats filled with zeros."""
        shape = [len(THRESHOLDS)] + list(image_shape)
        return cls(
            thresholds=THRESHOLDS,
            count_rain=np.zeros(shape),
            count_clear=np.zeros(shape),
            sum_baltrad=np.zeros(shape),
            sum_prx=np.zeros(shape),
            sum_square_baltrad=np.zeros(shape),
            sum_square_prx=np.zeros(shape),
            sum_product=np.zeros(shape),
            sum_diff_square=np.zeros(shape),
            true_detection=np.zeros(shape),
            false_detection=np.zeros(shape),
            product_tag="prx",
        )

    @classmethod
    def from_file(
        cls,
        stats_file: Path,
    ) -> "Stats":
        """Load stats from file."""
        product_tag, _ = stats_file.name.lower().split("_")
        data = xr.load_dataset(stats_file)
        return cls(
            thresholds=THRESHOLDS,
            count_rain=data["count_rain"].values,
            count_clear=data["count_clear"].values,
            sum_prx=data["sum_prx"].values,
            sum_baltrad=data["sum_baltrad"].values,
            sum_square_prx=data["sum_square_prx"].values,
            sum_square_baltrad=data["sum_square_baltrad"].values,
            sum_diff_square=data["sum_diff_square"].values,
            sum_product=data["sum_product"].values,
            true_detection=data["true_detection"].values,
            false_detection=data["false_detection"].values,
            product_tag=product_tag,
        )

    @property
    def as_dataset(self) -> xr.Dataset:
        """Transform stats to an xarray dataset."""
        return xr.Dataset(
            {
                "sum_baltrad": (("t", "y", "x"), self.sum_baltrad),
                "sum_square_baltrad": (("t", "y", "x"), self.sum_square_baltrad),
                "sum_prx": (("t", "y", "x"), self.sum_prx),
                "sum_square_prx": (("t", "y", "x"), self.sum_square_prx),
                "sum_diff_square": (("t", "y", "x"), self.sum_diff_square),
                "sum_product": (("t", "y", "x"), self.sum_product),
                "count_rain": (("t", "y", "x"), self.count_rain),
                "count_clear": (("t", "y", "x"), self.count_clear),
                "true_detection": (("t", "y", "x"), self.true_detection),
                "false_detection": (("t", "y", "x"), self.false_detection),
            },
            coords={"thresholds": ("t", self.thresholds)},
        )

    def to_netcdf(
        self,
        outdir: Path,
        product_tag: str,
    ) -> Path:
        """Write stats to netcdf file."""
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / f"{product_tag}_stats.nc"
        self.as_dataset.to_netcdf(outfile)
        return outfile

    @staticmethod
    def get_tag(prx_file: Path):
        """Get tag from given filename."""
        _, _, product, sensor, _, _ = prx_file.name.lower().split("_")
        return f"{product}-{sensor}"

    def update(
        self,
        attr: str,
        idx: int,
        filt: np.ndarray,
        new: int | float | np.ndarray,
    ) -> None:
        """Update attribute."""
        data = getattr(self, attr)
        data[idx] = np.where(filt, data[idx] + new, data[idx])

    def add_record(
        self,
        rr_prx: np.ndarray,
        rr_baltrad: np.ndarray,
    ) -> None:
        """Add given rainfall rate record to the stats."""
        for idx, threshold in enumerate(self.thresholds):

            baltrad_above_threshold = rr_baltrad >= threshold
            baltrad_below_threshold = rr_baltrad < threshold
            prx_above_threshold = rr_prx >= threshold
            prx_is_finite = np.isfinite(rr_prx)

            filt_rain = baltrad_above_threshold & prx_is_finite
            filt_clear = baltrad_below_threshold & prx_is_finite
            filt_true_detection = baltrad_above_threshold & prx_above_threshold
            filt_false_detection = baltrad_below_threshold & prx_above_threshold

            self.update("count_rain", idx, filt_rain, 1)
            self.update("count_clear", idx, filt_clear, 1)
            self.update("sum_baltrad", idx, filt_rain, rr_baltrad)
            self.update("sum_prx", idx, filt_rain, rr_prx)
            self.update("sum_square_baltrad", idx, filt_rain, rr_baltrad ** 2)
            self.update("sum_square_prx", idx, filt_rain, rr_prx ** 2)
            self.update("sum_product", idx, filt_rain, rr_prx * rr_baltrad)
            self.update("sum_diff_square", idx, filt_rain, (rr_baltrad - rr_prx) ** 2)
            self.update("true_detection", idx, filt_true_detection, 1)
            self.update("false_detection", idx, filt_false_detection, 1)

    def plot_stats(
        self,
        area: AreaDefinition,
        stats_dir: Path,
    ) -> None:
        """Plot statistics."""

        crs = area.to_cartopy_crs()

        for metric, values in self.metrics.items():
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
            outfile = stats_dir / f"{self.product_tag}_{metric}.png"
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
