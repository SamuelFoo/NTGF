import sqlite3
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from skimage import measure

from GKTH.constants import kB
from script_single_layer import (
    DATA_DIR,
    get_delta_vs_h,
    get_residuals_phase,
    read_residual_delta_database,
    read_residuals_delta_database_mev,
)

FIGURE_SIZE = (8, 4)

font = {"size": 12}
matplotlib.rc("font", **font)
matplotlib.rc("axes", **{"xmargin": 0})  # No padding on x-axis

PRESENTATION_MEDIA_DIR = Path("presentation_media")
PRESENTATION_MEDIA_DIR.mkdir(exist_ok=True)

REPORT_MEDIA_DIR = PRESENTATION_MEDIA_DIR / "report"
REPORT_MEDIA_DIR.mkdir(exist_ok=True)
SLIDES_MEDIA_DIR = PRESENTATION_MEDIA_DIR / "slides"
SLIDES_MEDIA_DIR.mkdir(exist_ok=True)

##############################
#   General plot functions   #
###############################


def plot_series_cmap(
    ax: Axes,
    plot_fn: Callable,
    series_list: List[Tuple],
    series_cmap_values: List[float],
    cmap_min: float,
    cmap_max: float,
    cmap: str = "copper",
):

    for series, series_cmap_value in zip(series_list, series_cmap_values):
        color = plt.get_cmap(cmap)(
            (series_cmap_value - cmap_min) / (cmap_max - cmap_min)
        )
        x, y = series
        plot_fn(x, y, color=color)

    sc = ax.scatter([], [], c=[], cmap=cmap, vmin=cmap_min, vmax=cmap_max)
    return sc


def plot_series_cmap_log_scale(
    ax: Axes, plot_fn: Callable, series_list, series_cmap_values, cmap_min, cmap_max
):
    for series, series_cmap_value in zip(series_list, series_cmap_values):
        # Apply logarithmic scaling to the color mapping
        log_min = np.log10(max(cmap_min, 1e-10))
        log_max = np.log10(max(cmap_max, 1e-10))
        log_value = np.log10(max(series_cmap_value, 1e-10))
        color = plt.cm.copper((log_value - log_min) / (log_max - log_min))
        x, y = series
        plot_fn(x, y, color=color)

    plot_norm = LogNorm(vmin=max(cmap_min, 1e-10), vmax=cmap_max)
    sc = ax.scatter([], [], c=[], cmap="copper", norm=plot_norm)
    return sc


def join_axes_with_shared_x(ax1: Axes, ax2: Axes):
    """Join two axes with a shared x-axis."""
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    ax2.set_position([pos2.x0, pos1.y0 - pos2.height, pos2.width, pos2.height])
    plt.setp(ax1.get_xticklabels(), visible=False)
    return ax1, ax2


def get_contour(
    x_mesh: np.ndarray, y_mesh: np.ndarray, z_mesh: np.ndarray, value: np.float64
):
    contours = measure.find_contours(z_mesh, value)
    contour = contours[0]
    x_scale = 1 / x_mesh.shape[1] * x_mesh.max()
    y_scale = 1 / y_mesh.shape[1] * y_mesh.max()
    return contour[:, 1] * x_scale, contour[:, 0] * y_scale


def move_axes(ax: Axes, x: float, y: float):
    pos = ax.get_position()
    ax.set_position([pos.x0 + x, pos.y0 + y, pos.width, pos.height])


def align_subplot_bottom(ax_to_align_to: Axes, axes_to_align: List[Axes]):
    # Sort axes_to_align by their y0 position
    axes_to_align.sort(key=lambda ax: ax.get_position().y0)

    # Get the gap_at_bottom between the axes to align to and the axes to align
    gap_at_bottom = (
        axes_to_align[0].get_position().y0 - ax_to_align_to.get_position().y0
    )

    # Add the gap evenly to all the axes to align
    n_axes = len(axes_to_align)
    for i in range(n_axes):
        ax = axes_to_align[-i - 1]
        pos = ax.get_position()
        ax.set_position(
            [
                pos.x0,
                pos.y0 - gap_at_bottom / n_axes * (i + 1),
                pos.width,
                pos.height + gap_at_bottom / n_axes,
            ]
        )


def align_subplot_left(ax_to_align_to: Axes, axes_to_align: List[Axes]):
    # Sort axes_to_align by their x0 position
    axes_to_align.sort(key=lambda ax: ax.get_position().x0)

    # Get the gap_at_left between the axes to align to and the axes to align
    gap_at_left = axes_to_align[0].get_position().x0 - ax_to_align_to.get_position().x0

    # Add the gap evenly to all the axes to align
    n_axes = len(axes_to_align)
    for i in range(n_axes):
        ax = axes_to_align[-i - 1]
        pos = ax.get_position()
        ax.set_position(
            [
                pos.x0 - gap_at_left / n_axes * (i + 1),
                pos.y0,
                pos.width + gap_at_left / n_axes,
                pos.height,
            ]
        )


def align_subplot_right(ax_to_align_to: Axes, axes_to_align: List[Axes]):
    # Sort axes_to_align by their x0 position
    axes_to_align.sort(key=lambda ax: ax.get_position().x1)

    # Get the gap_at_right between the axes to align to and the axes to align
    gap_at_right = (
        ax_to_align_to.get_position().x1 - axes_to_align[-1].get_position().x1
    )

    # Add the gap evenly to all the axes to align
    n_axes = len(axes_to_align)
    for i in range(n_axes):
        ax = axes_to_align[i]
        pos = ax.get_position()
        ax.set_position(
            [
                pos.x0 + gap_at_right / n_axes * i,
                pos.y0,
                pos.width + gap_at_right / n_axes,
                pos.height,
            ]
        )


####################
#   Single layer   #
####################


def plot_for_lambda(ax: Axes, _lambda: float):
    query = (
        "SELECT h, Delta, x_vals, residuals FROM results WHERE _lambda = ?",
        (_lambda,),
    )
    h_list_mev, Deltas_mev, plot_tuples_mev = read_residuals_delta_database_mev(query)
    min_h_mev = min(h_list_mev)
    max_h_mev = max(h_list_mev)

    plot_series_cmap(ax, ax.plot, plot_tuples_mev, h_list_mev, min_h_mev, max_h_mev)
    Delta_plot_tuples = [([Delta], [0]) for Delta in Deltas_mev]
    scatter_fn = lambda x, y, **kwargs: ax.scatter(x, y, marker="x", **kwargs)
    sc = plot_series_cmap(
        ax, scatter_fn, Delta_plot_tuples, h_list_mev, min_h_mev, max_h_mev
    )

    ax.scatter([], [], color="black", marker="x", label="Root")
    ax.axhline(y=0, color="gray", linestyle="--")
    ax.set_xlabel(r"$\Delta_0$ (meV)")
    ax.set_ylabel(r"$\delta \Delta$ (meV)")
    ax.set_title(rf"$\lambda = {_lambda}$")
    ax.legend()
    return sc


def plot_for_lambda_zeros(_lambda):
    h_list, Deltas = get_delta_vs_h(_lambda)

    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_subplot()

    sc = ax.scatter(
        [], [], c=[], cmap="copper", vmin=min(h_list) * 1e3, vmax=max(h_list) * 1e3
    )
    cbar = fig.colorbar(sc)
    cbar.set_label("h (meV)")

    for h, Delta in zip(h_list, Deltas):
        color = plt.cm.copper((h - min(h_list)) / (max(h_list) - min(h_list)))
        ax.scatter([Delta * 1e3], [0], color=color, marker="x")

    ax.scatter([], [], color="black", marker="x", label="Root")

    ax.axhline(y=0, color="gray", linestyle="--")
    ax.set_xlabel(r"$\Delta_0$ (meV)")
    ax.set_ylabel("Residual (meV)")
    ax.set_title(rf"$\lambda = {_lambda}$")
    ax.legend()
    return fig


def plot_for_lambda_h_list(_lambda, h_list):
    query = (
        "SELECT h, Delta, x_vals, residuals FROM results WHERE _lambda = ? AND h IN ({seq})".format(
            seq=",".join(["?"] * len(h_list))
        ),
        [_lambda] + h_list,
    )
    h_list, Deltas, plot_tuples = read_residual_delta_database(query)

    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_subplot()

    for h, (x_vals, residuals), Delta in zip(h_list, plot_tuples, Deltas):
        x_vals_meV = np.array(x_vals) * 1e3
        residuals_meV = np.array(residuals) * 1e3
        ax.plot(x_vals_meV, residuals_meV, label=f"h = {h * 1e3} meV")

    ax.axhline(y=0, color="gray", linestyle="--")
    ax.set_xlabel(r"$\Delta_0$ (meV)")
    ax.set_ylabel(r"$\delta \Delta$ (meV)")
    ax.set_title(rf"$\lambda = {_lambda}$")
    ax.legend()
    return fig


def plot_gap_h(ax: Axes, lambda_list: float):
    for _lambda in lambda_list:
        h, delta = get_delta_vs_h(_lambda)
        ax.plot(
            h * 1e3,
            delta * 1e3,
            linestyle="--",
            marker="o",
            label=rf"$\lambda = {_lambda}$",
        )

    ax.legend()
    ax.set_xlabel(r"$h$ (meV)")
    ax.set_ylabel(r"$\Delta_s$ (meV)")
    ax.set_ylim(0, None)


def plot_gap_h_log(ax: Axes, lambda_list: float):
    for _lambda in lambda_list:
        h, delta = get_delta_vs_h(_lambda)
        ax.plot(
            h * 1e3,
            delta * 1e3,
            linestyle="--",
            marker="o",
            label=rf"$\lambda = {_lambda}$",
        )

    ax.legend()

    # Log scale for x and y axes
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"$h$ (meV)")
    ax.set_ylabel(r"$\Delta_s$ (meV)")


def plot_lambda_h_report():
    fig = plt.figure(figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 4 * 0.7))

    gs = gridspec.GridSpec(4, 2)

    lambda_list = [0.0, 0.04, 0.08, 0.1, 0.15, 0.2]
    axes: List[Axes] = []
    cbars = []
    for i, _lambda in enumerate(lambda_list):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        sc = plot_for_lambda(ax, _lambda)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.legend().remove()
        ax.text(
            0.80,
            0.90,
            rf"$\lambda = {_lambda}$ ",
            transform=ax.transAxes,
            va="top",
            ha="center",
        )

        axes.append(ax)

        cbar = fig.colorbar(sc, ax=ax)
        cbars.append(cbar)

    x_offset = 0.04
    y_offset = 0.02
    y_offset_half = y_offset / 2
    for i, (ax, cbar) in enumerate(zip(axes, cbars)):
        if i // 2 == 0:
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0 + y_offset_half, pos.width, pos.height])
            pos = cbar.ax.get_position()
            cbar.ax.set_position(
                [pos.x0, pos.y0 + y_offset_half, pos.width, pos.height]
            )

        if i // 2 == 2:
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0 - y_offset_half, pos.width, pos.height])
            pos = cbar.ax.get_position()
            cbar.ax.set_position(
                [pos.x0, pos.y0 - y_offset_half, pos.width, pos.height]
            )

        if i % 2 == 0:
            pos = ax.get_position()
            ax.set_position([pos.x0 - x_offset, pos.y0, pos.width, pos.height])
            pos = cbar.ax.get_position()
            cbar.ax.set_position([pos.x0 - x_offset, pos.y0, pos.width, pos.height])

    sc = ax.scatter([], [], color="k", marker="x", label="root")
    fig.legend(
        handles=[sc],
        loc="upper center",
        ncol=1,
        fontsize=12,
        bbox_to_anchor=(0.45, 0.94),
        bbox_transform=fig.transFigure,
    )

    y_offset = 0.05
    ax = fig.add_subplot(gs[3, :])
    plot_gap_h_log(ax, [0.05, 0.1, 0.15, 0.2])
    align_subplot_left(axes[0], [ax])
    align_subplot_right(axes[-1], [ax])
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 - y_offset, pos.width, pos.height])

    fig.text(
        -0.02,
        0.59,
        r"$\delta \Delta$ (meV)",
        va="center",
        rotation="vertical",
        fontsize=12,
    )
    fig.text(
        0.93,
        0.59,
        r"$h$ (meV)",
        va="center",
        rotation="vertical",
        fontsize=12,
    )
    fig.text(0.45, 0.26, r"$\Delta_0$ (meV)", ha="center", fontsize=12)

    fig.text(-0.01, 0.9, "(a)", ha="center", fontsize=12)
    fig.text(-0.01, 0.24, "(b)", ha="center", fontsize=12)

    return fig


def plot_residual_phase(
    fig: Figure,
    ax: Axes,
    Delta_mesh_mev: NDArray,
    h_mesh_mev: NDArray,
    residual_mesh_mev: NDArray,
):
    bound = np.abs((residual_mesh_mev).max())
    normalize = Normalize(vmin=-bound, vmax=bound)

    ax.set_xlabel(r"$\Delta_0$ (meV)")
    ax.set_ylabel("h (meV)")

    sc = ax.contourf(
        Delta_mesh_mev,
        h_mesh_mev,
        residual_mesh_mev,
        cmap="bwr",
        norm=normalize,
        levels=100,
    )
    cbar = fig.colorbar(sc, label=r"$\delta \Delta$ (meV)")
    return cbar


def get_residual_phase_stability(
    Delta_mesh_mev: NDArray,
    h_mesh_mev: NDArray,
    residual_mesh_mev: NDArray,
):
    """Get the residual phase stability from the residual mesh.
    Args:
        Delta_mesh_mev (NDArray): Delta mesh in meV.
        h_mesh_mev (NDArray): h mesh in meV.
        residual_mesh_mev (NDArray): Residual mesh in meV.
    Returns:
        Stable and unstable zeros of the self-consistency equation.
    """
    zeros_x, zeros_y = get_contour(Delta_mesh_mev, h_mesh_mev, residual_mesh_mev, 0.0)

    # Remove ill-defined points at x = 0
    zeros_y = zeros_y[zeros_x != 0]
    zeros_x = zeros_x[zeros_x != 0]

    # Sort by x coord
    sort_idxs = np.argsort(zeros_x)
    zeros_x = zeros_x[sort_idxs]
    zeros_y = zeros_y[sort_idxs]

    zeros_grad = np.gradient(zeros_y, zeros_x)
    (neg_grad_idxs,) = np.where(zeros_grad <= 0)

    # Midpoint idx is the first index such that more than 90% of subsequent gradients are negative
    midpoint = neg_grad_idxs[
        np.argmax(np.diff(neg_grad_idxs) > 0.9 * len(neg_grad_idxs))
    ]
    stable_zeros = (zeros_x[midpoint:], zeros_y[midpoint:])
    unstable_zeros = (zeros_x[: midpoint + 1], zeros_y[: midpoint + 1])
    return stable_zeros, unstable_zeros


def plot_residual_phase_stability(
    ax: Axes,
    Delta_mesh_mev: NDArray,
    h_mesh_mev: NDArray,
    residual_mesh_mev: NDArray,
):
    stable_zeros, unstable_zeros = get_residual_phase_stability(
        Delta_mesh_mev, h_mesh_mev, residual_mesh_mev
    )

    ax.plot(*stable_zeros, color="k", label="Stable")
    ax.plot(*unstable_zeros, color="k", linestyle="--", label="Unstable")
    ax.legend()


def plot_residual_phase_stability_report():
    def get_zeroes():
        tuple_list = [
            (0.1, 1e-3, 2e-3),
            (0.11, 5e-3, 5e-3),
            (0.12, 5e-3, 5e-3),
            (0.13, 1e-2, 1e-2),
            (0.14, 2e-2, 2e-2),
            (0.15, 2e-2, 2e-2),
            (0.16, 3e-2, 3e-2),
            (0.17, 3e-2, 3e-2),
            (0.18, 4e-2, 4e-2),
            (0.19, 4e-2, 4e-2),
            (0.20, 5e-2, 5e-2),
        ]
        tuple_list = np.array(tuple_list)
        N = 41

        stable_zeroes_list = []
        unstable_zeroes_list = []

        for i, (_lambda, h_end, max_Delta) in enumerate(tuple_list):
            Delta_mesh_mev, h_mesh_mev, residual_mesh_mev = get_residuals_phase(
                _lambda, max_Delta, h_end, N
            )
            stable_zeros, unstable_zeros = get_residual_phase_stability(
                Delta_mesh_mev, h_mesh_mev, residual_mesh_mev
            )

            stable_zeros = np.array(stable_zeros)
            unstable_zeros = np.array(unstable_zeros)
            max_values = np.concatenate(
                [
                    stable_zeros.max(axis=1, keepdims=True),
                    unstable_zeros.max(axis=1, keepdims=True),
                ],
                axis=1,
            )
            max_val_kx = max_values.max(axis=1, keepdims=True)
            stable_zeros /= max_val_kx
            unstable_zeros /= max_val_kx

            stable_zeroes_list.append(stable_zeros)
            unstable_zeroes_list.append(unstable_zeros)

        return tuple_list, stable_zeroes_list, unstable_zeroes_list

    tuple_list, stable_zeroes_list, unstable_zeroes_list = get_zeroes()
    _lambdas = tuple_list[:, 0]
    cmap = "copper"

    fig = plt.figure(
        figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 3),
    )

    # Top plot
    ax1 = fig.add_subplot(311)
    sc = plot_series_cmap(
        ax=ax1,
        plot_fn=ax1.plot,
        series_list=stable_zeroes_list,
        series_cmap_values=_lambdas,
        cmap_min=min(_lambdas),
        cmap_max=max(_lambdas),
        cmap=cmap,
    )
    plot_series_cmap(
        ax=ax1,
        plot_fn=lambda *args, **kwargs: ax1.plot(*args, linestyle="--", **kwargs),
        series_list=unstable_zeroes_list,
        series_cmap_values=_lambdas,
        cmap_min=min(_lambdas),
        cmap_max=max(_lambdas),
        cmap=cmap,
    )

    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel(r"$\Delta_0/\max(\Delta_0)$")
    ax1.set_ylabel(r"$h/\max(h)$")

    # Middle plot
    Delta_sample = 0.2
    h_samples = []
    _lambda_samples = []
    for _lambda, stable_zeroes in zip(_lambdas, unstable_zeroes_list):
        x, y = stable_zeroes
        if Delta_sample < x.min() or Delta_sample > y.max():
            continue

        interp1d_func = interp1d(x, y)
        h_sample = interp1d_func(Delta_sample)
        h_samples.append(h_sample)
        _lambda_samples.append(_lambda)

    ax2 = fig.add_subplot(312)
    ax2.plot(_lambda_samples, h_samples, zorder=-1, color="k")
    ax2.set_ylabel(r"$h/\max(h)$")

    plot_series_cmap(
        ax=ax2,
        plot_fn=ax2.scatter,
        series_list=list(zip(_lambda_samples, h_samples)),
        series_cmap_values=_lambda_samples,
        cmap_min=min(_lambdas),
        cmap_max=max(_lambdas),
        cmap=cmap,
    )

    ax1.scatter([0.2] * len(_lambda_samples), h_samples, color="k", zorder=100, s=15)
    ax1.axvline(0.2, color="k", zorder=99)

    # Bottom plot
    ax3 = fig.add_subplot(313)

    critical_pts = []
    for unstable_zeroes in unstable_zeroes_list:
        x, y = unstable_zeroes
        max_idx = np.argmax(x)
        critical_pts.append((x[max_idx], y[max_idx]))

    series_list = [
        ([_lambda], [Delta_c]) for _lambda, (Delta_c, _) in zip(_lambdas, critical_pts)
    ]
    ax3.plot(_lambdas, [pt[0] for pt in critical_pts], zorder=-1, color="k")
    plot_series_cmap(
        ax3, ax3.scatter, series_list, _lambdas, min(_lambdas), max(_lambdas), cmap=cmap
    )
    ax3.set_xlabel(r"$\lambda$ (meV)")
    ax3.set_ylabel(r"$\Delta_c$ (meV)")

    for _lambda, (Delta_c, h_c) in zip(_lambdas, critical_pts):
        ax1.plot(
            [Delta_c, Delta_c], [0, h_c], linestyle="--", color="lightgray", zorder=-1
        )

    fig.subplots_adjust(hspace=0.4)
    join_axes_with_shared_x(ax2, ax3)

    ax1.text(0.05, 0.10, "(a)", transform=ax1.transAxes, va="center", ha="center")
    ax2.text(0.95, 0.90, "(b)", transform=ax2.transAxes, va="center", ha="center")
    ax3.text(0.05, 0.90, "(c)", transform=ax3.transAxes, va="center", ha="center")

    fig.colorbar(sc, ax=[ax1, ax2, ax3], label=r"$\lambda$ (meV)")

    return fig


################
#   Junction   #
################


def get_current_angle_data(layers_str: str, tunneling: float):
    """Get the current angle data from the database.

    Args:
        layers_str (str): String representation of the layers. E.g., "S1_N_S2"
        tunneling (float): Tunneling parameter.

    Returns:
        df: DataFrame with columns:
            - temperature (in Kelvin)
            - tunneling (in meV)
            - jc (critical current density in A/m^2)
            - phase (in rad)
    """
    db_name = f"{layers_str}_current.db"
    conn = sqlite3.connect(DATA_DIR / "current" / db_name)
    query = f"SELECT temperature, tunneling, jc, phase FROM current WHERE tunneling = {tunneling}"
    df = pd.read_sql_query(query, conn)
    df["temperature"] = df["temperature"] / kB
    df["jc"] = df["jc"] / 1e6

    sorted_idxs = np.argsort(df["phase"])
    sorted_df = df.iloc[sorted_idxs]

    return sorted_df


def get_current_angle_subplot(ax: Axes, layers_str: str, tunneling: float):
    df = get_current_angle_data(layers_str, tunneling)

    plot_tuples = []
    temperatures = []
    for i, temperature in enumerate(df["temperature"].unique()):
        subset = df[df["temperature"] == temperature]
        plot_tuples.append((subset["phase"], subset["jc"]))
        temperatures.append(temperature)

    sc = plot_series_cmap_log_scale(
        ax, ax.plot, plot_tuples, temperatures, min(temperatures), max(temperatures)
    )

    ax.set_xlabel(r"$\phi$ (rad)")
    ax.set_ylabel(r"$j$ $(M A\ m^{-2})$")
    ax.set_xlim(-np.pi, np.pi)

    x_ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

    return sc


def get_critical_current_data(layers_str: str, tunneling: float):
    """Get the critical current data from the database.

    Args:
        layers_str (str): String representation of the layers. E.g., "S1_N_S2"
        tunneling (float): Tunneling parameter.

    Returns:
        df: DataFrame sorted by temperature with columns:
            - temperature (in Kelvin)
            - tunneling (in meV)
            - jc (critical current density in A/m^2)
            - phase (in rad)
    """
    db_name = f"{layers_str}_critical.db"
    conn = sqlite3.connect(DATA_DIR / "critical_current" / db_name)
    query = f"SELECT temperature, tunneling, jc, phase FROM current WHERE tunneling = {tunneling}"
    df = pd.read_sql_query(query, conn)
    df["temperature"] = df["temperature"] / kB
    df["jc"] = df["jc"] / 1e6

    sorted_idxs = np.argsort(df["temperature"])
    sorted_df = df.iloc[sorted_idxs]

    return sorted_df


def get_critical_current_subplot(ax: Axes, layers_str: str, tunneling: float):
    df = get_critical_current_data(layers_str, tunneling)

    plot_tuples = []
    temperatures = []
    for temperature in df["temperature"].unique():
        subset = df[df["temperature"] == temperature]
        plot_tuples.append(([temperature], subset["jc"]))
        temperatures.append(temperature)

    scatter_fn = lambda *args, **kwargs: ax.scatter(*args, **kwargs, s=10)
    sc = plot_series_cmap_log_scale(
        ax, scatter_fn, plot_tuples, temperatures, min(temperatures), max(temperatures)
    )
    ax.plot(df["temperature"], df["jc"], color="k", zorder=-1)

    ax.set_xlabel(r"$T$ (K)")
    ax.set_ylabel(r"$j_c$ $(M A\ m^{-2})$")
    ax.set_xlim(0, 12)
    ax.set_ylim(0.1, None)

    return sc


def get_critical_phase_subplot(ax: Axes, layers_str: str, tunneling: float):
    df = get_critical_current_data(layers_str, tunneling)

    plot_tuples = []
    temperatures = []
    for temperature in df["temperature"].unique():
        subset = df[df["temperature"] == temperature]
        plot_tuples.append(([temperature], subset["phase"]))
        temperatures.append(temperature)

    scatter_fn = lambda *args, **kwargs: ax.scatter(*args, **kwargs, s=10)
    sc = plot_series_cmap_log_scale(
        ax, scatter_fn, plot_tuples, temperatures, min(temperatures), max(temperatures)
    )
    ax.plot(df["temperature"], df["phase"], color="k", zorder=-1)

    ax.set_xlabel(r"$T$ (K)")
    ax.set_ylabel(r"$\phi_c$ (rad)")
    ax.set_xlim(0, 12)
    y_ticks = [0, np.pi / 2, np.pi]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$"])

    return sc


def plot_critical_current(layers_str: str, tunneling: float):
    fig = plt.figure(figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1]))

    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1])  # 2 rows, 2 columns
    ax1 = fig.add_subplot(gs[:, 0])  # Left column spanning both rows
    ax2 = fig.add_subplot(gs[0, 1])  # Top-right
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)  # Bottom-right

    join_axes_with_shared_x(ax2, ax3)
    align_subplot_bottom(ax1, [ax2, ax3])

    pos1 = ax1.get_position()
    ax1.set_position([pos1.x0 - 0.1, pos1.y0, pos1.width, pos1.height])

    # Add labels to the plots
    ax1.text(0.10, 0.95, "(a)", transform=ax1.transAxes, va="top", ha="center")
    ax2.text(0.9, 0.95, "(b)", transform=ax2.transAxes, va="top", ha="center")
    ax3.text(0.10, 0.95, "(c)", transform=ax3.transAxes, va="top", ha="center")

    get_current_angle_subplot(ax1, layers_str, tunneling)
    sc = get_critical_current_subplot(ax2, layers_str, tunneling)
    get_critical_phase_subplot(ax3, layers_str, tunneling)

    ax2.set_xlabel("")

    axes = [ax1, ax2, ax3]
    cbar = fig.colorbar(sc, ax=axes)
    cbar.set_label(r"$T$ (K)")
    tick_values = [1.0, 2.0, 4.0, 8.0]
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{t:.1f}" for t in tick_values])

    return fig


def plot_current_diff_tunneling():
    fig = plt.figure(figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 2))

    ax1 = fig.add_subplot(221)
    for t in np.array([0.5, 1, 2]) * 1e-3:
        df = get_critical_current_data("S1_N_S2", t)
        ax1.plot(df["temperature"], df["jc"], label=f"$t = {t * 1e3:.2f}$ meV")

    ax1.legend()
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, None)
    ax1.set_ylabel(r"$j_c$ $(M A\ m^{-2})$")

    series_list = []
    tunneling_params = np.array([0.1, 0.25, 0.5, 1, 2, 5, 10]) * 1e-3
    for t in tunneling_params:
        df = get_critical_current_data("S1_N_S2", t)
        series_list.append((df["temperature"], df["jc"] / max(df["jc"])))

    ax2 = fig.add_subplot(223, sharex=ax1)
    join_axes_with_shared_x(ax1, ax2)

    ax2.set_xlabel(r"$T$ (K)")
    ax2.set_ylabel(r"$j_c / j_{c0}$")
    ax2.set_ylim(0, 1)
    sc = plot_series_cmap_log_scale(
        ax=ax2,
        plot_fn=ax2.plot,
        series_list=series_list,
        series_cmap_values=tunneling_params,
        cmap_min=min(tunneling_params),
        cmap_max=max(tunneling_params),
    )

    second_col_offset = 0.08

    ax3 = fig.add_subplot(222)
    pos3 = ax3.get_position()
    ax3.set_position([pos3.x0 + second_col_offset, pos3.y0, pos3.width, pos3.height])
    for t in np.array([0.5, 1, 2]) * 1e-3:
        df = get_current_angle_data("S1_N_S2", t)
        df: pd.DataFrame = df[df["temperature"] == df["temperature"].min()]
        ax3.plot(
            df["phase"],
            df["jc"] / df["jc"].abs().max(),
            label=f"$t = {t * 1e3:.2f}$ meV",
        )
    ax3.set_xlabel(r"$\phi$ (rad)")
    x_ticks = [-np.pi, 0, np.pi]
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    ax3.set_ylabel(r"$j / j_0$")
    ax3.set_ylim(-1, 1)
    ax3.legend()

    ax4 = fig.add_subplot(224)
    pos4 = ax4.get_position()
    ax4.set_position([pos4.x0 + second_col_offset, pos4.y0, pos4.width, pos4.height])
    critical_phases = []
    for t in tunneling_params:
        df = get_critical_current_data("S1_N_S2", t)
        critical_phases.append(df["phase"].max())

    series_list = list(
        zip(tunneling_params[:, np.newaxis], np.array(critical_phases)[:, np.newaxis])
    )
    plot_series_cmap_log_scale(
        ax=ax4,
        plot_fn=ax4.scatter,
        series_list=series_list,
        series_cmap_values=tunneling_params,
        cmap_min=min(tunneling_params),
        cmap_max=max(tunneling_params),
    )
    ax4.plot(tunneling_params, critical_phases, color="k", zorder=-1)

    ax4.axhline(y=np.pi / 2, color="gray", linestyle="--")
    y_ticks = [0, np.pi / 2, np.pi]
    ax4.set_yticks(y_ticks)
    ax4.set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$"])
    ax4.set_xlabel(r"$t$ (meV)")
    ax4.set_ylabel(r"$\phi_{c0}$ (rad)")
    ax4.set_xscale("log")

    align_subplot_bottom(ax4, [ax1, ax2])

    # Add labels to the plots
    ax1.text(0.10, 0.95, "(a)", transform=ax1.transAxes, va="top", ha="center")
    ax2.text(0.90, 0.95, "(b)", transform=ax2.transAxes, va="top", ha="center")
    ax3.text(0.10, 0.95, "(c)", transform=ax3.transAxes, va="top", ha="center")
    ax4.text(0.10, 0.95, "(d)", transform=ax4.transAxes, va="top", ha="center")

    cbar = plt.colorbar(sc, ax=[ax1, ax2, ax3, ax4])
    cbar.set_label(r"$t$ (meV)")
    return fig
