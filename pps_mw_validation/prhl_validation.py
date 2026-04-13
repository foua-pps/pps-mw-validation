from dataclasses import dataclass
from pathlib import Path

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

    @property
    def metrics(self) -> dict[str, np.ndarray]:
        """Get the main metrics."""
        prx_mean = self.sum_prx / self.count_rain
        prx_square_mean = self.sum_square_prx / self.count_rain
        prx_std = np.sqrt(prx_square_mean - prx_mean ** 2)

        baltrad_mean = self.sum_baltrad / self.count_rain
        baltrad_square_mean = self.sum_square_baltrad / self.count_rain
        baltrad_std = np.sqrt(baltrad_square_mean - baltrad_mean ** 2)

        pod = self.true_detection / self.count_rain
        pofd = self.false_detection / self.count_clear
        covariance = self.sum_product / self.count_rain - prx_mean * baltrad_mean
        rmse = np.sqrt(self.sum_diff_square / self.count_rain)

        return {
            "bias": prx_mean - baltrad_mean,
            "fse": rmse / baltrad_mean,
            "pod": pod,
            "pofd": pofd,
            "corr":  covariance / (prx_std * baltrad_std)
        }

    @property
    def validation_score(self) -> dict[str, dict[float, tuple[float, float]]]:
        """Get the validation score in a json friendly format."""
        return {
            metric: {
                threshold: (_mean, _std) for threshold, _mean, _std in zip(
                    self.thresholds,
                    np.nanmean(np.where(np.isfinite(values), values, np.nan), axis=(1, 2)),
                    np.nanstd(np.where(np.isfinite(values), values, np.nan), axis=(1, 2)),
                )
            } for metric, values in self.metrics.items()
        }

    @staticmethod
    def get_tag(prx_file: Path):
        """Get tag from given filename."""
        _, _, product, sensor, _, _ = prx_file.name.lower().split("_")
        return f"{product}-{sensor}"

    def add(
        self,
        rr_prx: np.ndarray,
        rr_baltrad: np.ndarray
    ) -> None:
        """Add given rainfall rate record to the stats."""
        for idx, threshold in enumerate(self.thresholds):

            filt_rain = (rr_baltrad >= threshold) & np.isfinite(rr_prx)
            filt_clear = (rr_baltrad < threshold) & np.isfinite(rr_prx)

            for filt, data in [(filt_rain, self.count_rain), (filt_clear, self.count_clear)]:
                data[idx] = np.where(filt, data[idx] + 1, data[idx])

            for data, rr in [(self.sum_baltrad, rr_baltrad), (self.sum_prx, rr_prx)]:
                data[idx] = np.where(filt_rain, data[idx] + rr, data[idx])

            for data, rr in [(self.sum_square_baltrad, rr_baltrad), (self.sum_square_prx, rr_prx)]:
                data[idx] = np.where(filt_rain, data[idx] + rr ** 2, data[idx])

            self.sum_product[idx] = np.where(
                filt_rain,
                self.sum_product[idx] + rr_prx * rr_baltrad,
                self.sum_product[idx],
            )
            self.sum_diff_square[idx] = np.where(
                filt_rain,
                self.sum_diff_square[idx] + (rr_baltrad - rr_prx) ** 2,
                self.sum_diff_square[idx]
            )
            self.true_detection[idx] = np.where(
                (rr_baltrad >= threshold) & (rr_prx >= threshold),
                self.true_detection[idx] + 1,
                self.true_detection[idx]
            )
            self.false_detection[idx] = np.where(
                (rr_baltrad < threshold) & (rr_prx >= threshold),
                self.false_detection[idx] + 1,
                self.false_detection[idx]
            )

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
