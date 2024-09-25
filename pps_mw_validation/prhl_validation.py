from pathlib import Path
import logging
import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import xarray as xr  # type: ignore


CLIP_VALUE_MIN = 1  # mm / h
CLIP_VALUE_MAX = 1e9  # mm / h
SURFACE_TYPE = {"land": 1, "ocean": 2}
FIG_SIZE = (9, 7)


logger = logging.getLogger(__name__)


def get_stats(
    prhl_data: xr.Dataset,
    baltrad: np.ndarray,
    min_rr: float,
) -> xr.Dataset:
    """Get stats from comparison of a single scene."""
    condition = prhl_data["condition"].values
    prhl = prhl_data["rainfall_rate"].values
    datasets = []
    for surface_type, value in SURFACE_TYPE.items():
        filt_finite = np.isfinite(baltrad) & np.isfinite(prhl)
        filt_surf = np.bitwise_and(condition, value) == value
        detection = (prhl >= min_rr) & filt_finite & filt_surf
        no_detection = (prhl < min_rr) & filt_finite & filt_surf
        true_positive = (
            (prhl >= min_rr)
            & (baltrad >= min_rr)
            & filt_finite
            & filt_surf
        )
        dataset = xr.Dataset(
            {
                "baltrad": ("tp", baltrad[true_positive].flatten()),
                "prhl": ("tp", prhl[true_positive].flatten()),
                "condition": ("tp", condition[true_positive].flatten()),
            },
            attrs={
                f"n_true_positive_{surface_type}": np.count_nonzero(
                    baltrad[detection] >= min_rr
                ),
                f"n_false_positive_{surface_type}": np.count_nonzero(
                    baltrad[detection] < min_rr
                ),
                f"n_true_negative_{surface_type}": np.count_nonzero(
                    baltrad[no_detection] < min_rr
                ),
                f"n_false_negative_{surface_type}": np.count_nonzero(
                    baltrad[no_detection] >= min_rr
                ),
            }
        )
        datasets.append(dataset)
    dataset = xr.concat(datasets, dim="tp", combine_attrs="no_conflicts")
    return dataset


def get_summary_stats(
    baltrad: np.ndarray,
    prhl: np.ndarray,
) -> dict[str, float]:
    """Get summary stats."""
    clipped = np.clip(baltrad, a_min=CLIP_VALUE_MIN, a_max=CLIP_VALUE_MAX)
    return {
        "mean_relative_difference_percent": np.mean(
            100 * np.abs(prhl - baltrad) / clipped
        ),
        "std_relative_difference_percent": np.std(
            100 * (prhl - baltrad) / clipped
        ),
        "mean_difference_mm_per_hour": np.mean(prhl - baltrad),
        "std_difference_mm_per_hour": np.std(prhl - baltrad),
        "prhl_mean_mm_per_hour": np.mean(prhl),
        "baltrad_mean_mm_per_hour": np.mean(baltrad),
    }


def get_detection_performance(
    records: list[dict[str, int]]
) -> dict[str, dict[str, int | float]]:
    """Get detection performance."""
    performance = {}
    for surf_type in SURFACE_TYPE:
        stats: dict[str, int | float] = {
            p: int(np.sum([r[f"{p}_{surf_type}"] for r in records]))
            for p in [
                "n_true_positive",
                "n_false_positive",
                "n_true_negative",
                "n_false_negative"
            ]
        }
        pod, pofd = get_pod_and_pofd(**stats)
        stats["probability_of_detection"] = pod
        stats["probability_of_false_detection"] = pofd
        performance[surf_type] = stats
    return performance


def get_pod_and_pofd(
    n_true_positive,
    n_false_positive,
    n_true_negative,
    n_false_negative,
) -> tuple[float, float]:
    """Get POD and POFD from contingency table data."""
    pod = n_true_positive / (n_true_positive + n_false_negative)
    pofd = n_false_positive / (n_false_positive + n_true_negative)
    return pod, pofd


def make_plots(
    baltrad: np.ndarray,
    prhl: np.ndarray,
    tag: str,
    outpath: Path,
) -> None:
    """Make relative difference plots."""
    make_heatmap(baltrad, prhl, tag, outpath)
    make_relative_difference_plot(baltrad, prhl, tag, outpath)


def make_relative_difference_plot(
    baltrad: np.ndarray,
    prhl: np.ndarray,
    tag: str,
    outpath: Path,
) -> None:
    """Make relative difference plots."""
    n_bins = 50
    min_rr = 0.1  # [mm / h]
    max_rr = 10  # [mm / h]
    percentiles = [16, 50, 84]
    limits = np.logspace(
        np.log10(min_rr),
        np.log10(max_rr),
        n_bins + 1,
    )
    center = 10 ** (
        (np.log10(limits[0:-1]) + np.log10(limits[1::]))
        / 2
    )
    diff = prhl - baltrad

    if not outpath.exists():
        outpath.mkdir(parents=True, exist_ok=True)

    for plot in range(2):
        if plot == 0:
            mean = (baltrad + prhl) / 2
            xlabel = "Precipitation rate, (PRHL + BALTRAD) / 2, [mm / h]"
        else:
            mean = baltrad
            xlabel = "Precipitation rate,  BALTRAD, [mm / h]"

        data = np.full((n_bins, len(percentiles)), np.nan)

        for i in range(n_bins):
            filt = (mean >= limits[i]) & (mean < limits[i + 1])
            try:
                data[i] = np.percentile(100 * (diff / mean)[filt], percentiles)
            except IndexError:
                pass
        plt.figure(figsize=FIG_SIZE)
        for idx, p in enumerate(percentiles):
            plt.semilogx(center, data[:, idx], label=f"{p} %", linewidth=2)
            plt.grid(True)
            plt.legend()
            plt.ylabel("Relative difference, PRHL - BALTRAD, [%]")
            plt.xlabel(xlabel)
            plt.ylim([-200, 200])
            plt.xlim([min_rr, max_rr])
        outfile = outpath / f"phrl_baltrad_relative_diff_{tag}.png"
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
        logger.info(f"Wrote image file to disk: {outfile.as_posix()}")


def make_heatmap(
    baltrad: np.ndarray,
    prhl: np.ndarray,
    tag: str,
    outpath: Path,
) -> None:
    """Make a rain rate heatmap."""
    min_rr = 0.1  # [mm / h]
    max_rr = 50  # [mm / h]
    n_bins = 50
    n_colors = 32
    limits = np.logspace(
        np.log10(min_rr),
        np.log10(max_rr),
        n_bins + 1,
    )
    center = 10 ** (
        (np.log10(limits[0:-1]) + np.log10(limits[1::]))
        / 2
    )
    x, y = np.meshgrid(center, center)

    heatmap = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        for j in range(n_bins):
            heatmap[i, j] = np.count_nonzero(
                (baltrad >= limits[i])
                & (baltrad < limits[i + 1])
                & (prhl >= limits[j])
                & (prhl < limits[j + 1])
            )

    cmap = plt.get_cmap('jet', n_colors)
    plt.figure(figsize=FIG_SIZE)
    vmax = 100 * heatmap.max() // 100
    norm = matplotlib.colors.LogNorm(vmin=1, vmax=vmax)
    plt.scatter(x, y, s=40, c=heatmap.T, cmap=cmap, norm=norm)
    plt.plot(limits, limits, "-k", linewidth=2, label="1-to-1")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    cbar = plt.colorbar()
    cbar.set_label('Number of hits [-]', rotation=270)
    plt.xlabel("BALTRAD precipitation [mm / h]")
    plt.ylabel("PR-HL precipitation [mm / h]")
    if not outpath.exists():
        outpath.mkdir(parents=True, exist_ok=True)
    outfile = outpath / f"prhl_baltrad_scatter_{tag}.png"
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()
    logger.info(f"Wrote image file to disk: {outfile.as_posix()}")
