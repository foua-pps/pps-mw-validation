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


def get_surface_mask(condition: np.ndarray) -> dict[str, np.ndarray]:
    """Get surface mask."""
    return {
        surf_type: np.bitwise_and(condition, value) == value
        for surf_type, value in SURFACE_TYPE.items()
    } | {
         "all_surface": np.ones_like(condition, dtype=bool)
    }


def get_rain_rate_difference(
    baltrad: np.ndarray,
    prhl: np.ndarray,
    condition: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Get rain rate difference stats."""
    diff = prhl - baltrad
    clipped = np.clip(baltrad, a_min=CLIP_VALUE_MIN, a_max=CLIP_VALUE_MAX)
    return {
        surf_type: {
            "mean_relative_difference_percent": np.mean(
                100 * np.abs(diff[m]) / clipped[m]
            ),
            "std_relative_difference_percent": np.std(
                100 * diff[m] / clipped[m]
            ),
            "mean_difference_mm_per_hour": np.mean(diff[m]),
            "std_difference_mm_per_hour": np.std(diff[m]),
            "prhl_mean_mm_per_hour": np.mean(prhl[m]),
            "baltrad_mean_mm_per_hour": np.mean(baltrad[m]),
        } for surf_type, m in get_surface_mask(condition).items()
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


def plot_scene(
    prhl_data: xr.Dataset,
    baltrad: np.ndarray,
    min_rr: float,
    outpath: Path,
) -> None:
    """Plot the scene."""
    plot_rain_rate(prhl_data, baltrad, outpath)
    plot_detection(prhl_data["rainfall_rate"].data, baltrad, min_rr, outpath)


def plot_detection(
    prhl: np.ndarray,
    baltrad: np.ndarray,
    min_rr: float,
    outpath: Path,
) -> None:
    """Plot the detection performance of PR-HL."""
    filt_finite = np.isfinite(prhl) & np.isfinite(baltrad)
    filt_true_positive = (prhl >= min_rr) & (baltrad >= min_rr) & filt_finite
    filt_false_positive = (prhl >= min_rr) & (baltrad < min_rr) & filt_finite
    filt_true_negative = (prhl < min_rr) & (baltrad < min_rr) & filt_finite
    filt_false_negative = (prhl < min_rr) & (baltrad >= min_rr) & filt_finite

    r = np.zeros_like(prhl, dtype=int)
    r[filt_true_positive] = 1
    r[filt_true_negative] = 2
    r[filt_false_positive] = 3
    r[filt_false_negative] = 4

    clipped = np.clip(baltrad, a_min=CLIP_VALUE_MIN, a_max=CLIP_VALUE_MAX)
    diff = (prhl - baltrad) / clipped

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))

    # detection performance subplot
    n_colors = 5
    cmap = plt.get_cmap('jet', n_colors)
    lcmap = [cmap(i) for i in range(cmap.N)]
    lcmap[0] = (.5, .5, .5, 1.0)
    cmap = cmap.from_list('custom cmap', lcmap, n_colors)

    im = axes[0].imshow(r, cmap=cmap, vmin=0, vmax=5)
    axes[0].title.set_text("Detection of rain rate > 0.1 mm / h")
    cbar_ax = fig.add_axes([0.15, 0.08, 0.32, 0.05])
    cbar = fig.colorbar(
        im, cax=cbar_ax, orientation='horizontal', ticks=[1.5, 2.5, 3.5, 4.5]
    )
    cbar.ax.set_xticklabels(
        ['true positive', 'true negative', 'false positive', 'false negative'],
        rotation=15,
    )
    axes[0].set_yticklabels([])
    axes[0].set_xticklabels([])

    # reltive difference subplot
    cmap = plt.get_cmap('coolwarm', 17)
    diff[~filt_true_positive] = 0.0
    im = axes[1].imshow(100 * diff, cmap=cmap, vmin=-100, vmax=100)
    axes[1].title.set_text("PR-HL - BALTRAD")
    axes[1].set_yticklabels([])
    axes[1].set_xticklabels([])

    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.6, 0.08, 0.32, 0.05])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.ax.set_xlabel('Relative difference [%]')

    if not outpath.exists():
        outpath.mkdir(parents=True, exist_ok=True)
    outfile = outpath / "prhl_baltrad_detection_performance.png"
    plt.savefig(outfile,  bbox_inches='tight')
    plt.close()
    logger.info(f"Wrote image file to disk: {outfile.as_posix()}")


def plot_rain_rate(
    prhl_data: xr.Dataset,
    baltrad: np.ndarray,
    outpath: Path,
) -> None:
    """Plot the retrieved rain rate."""
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 8))
    n_colors = 64
    cmap = plt.get_cmap('jet', n_colors)
    lcmap = [cmap(i) for i in range(cmap.N)]
    lcmap[0] = (.5, .5, .5, 1.0)  # make first color gray
    cmap = cmap.from_list('custom cmap', lcmap, n_colors)
    vmin = 0
    vmax = np.ceil(np.nanmax(prhl_data["rainfall_rate"].values))
    for idx, (data, title) in enumerate(
        [
            (prhl_data["rainfall_rate"], "PR-HL"),
            (prhl_data["rainfall_rate_uncertainty"], "PR-HL uncertainty"),
            (baltrad, "BALTRAD"),
        ]
    ):
        im = axes[idx].imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[idx].title.set_text(title)
        axes[idx].set_yticklabels([])
        axes[idx].set_xticklabels([])

    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.35, 0.2, 0.38, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.ax.set_xlabel('Precipitation rate [mm / h]')

    if not outpath.exists():
        outpath.mkdir(parents=True, exist_ok=True)
    outfile = outpath / "prhl_baltrad_retrieved_rain_rate.png"
    plt.savefig(outfile,  bbox_inches='tight')
    plt.close()
    logger.info(f"Wrote image file to disk: {outfile.as_posix()}")


def make_plots(
    baltrad: np.ndarray,
    prhl: np.ndarray,
    condition: np.ndarray,
    outpath: Path,
) -> None:
    """Make plots."""
    for surf_type, m in get_surface_mask(condition).items():
        make_heatmap(baltrad[m], prhl[m], surf_type, outpath)
        make_relative_difference_plot(baltrad[m], prhl[m], surf_type, outpath)


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
        outfile = outpath / f"prhl_baltrad_relative_diff_{tag}_{plot}.png"
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
