import sqlite3
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from numpy.typing import NDArray
from skimage import measure

from GKTH.constants import kB
from script_single_layer import (
    DATA_DIR,
    get_delta_vs_h,
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
):
    for series, series_cmap_value in zip(series_list, series_cmap_values):
        color = plt.cm.copper((series_cmap_value - cmap_min) / (cmap_max - cmap_min))
        x, y = series
        plot_fn(x, y, color=color)

    sc = ax.scatter([], [], c=[], cmap="copper", vmin=cmap_min, vmax=cmap_max)
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


####################
#   Single layer   #
####################


def plot_for_lambda(_lambda):
    query = (
        "SELECT h, Delta, x_vals, residuals FROM results WHERE _lambda = ?",
        (_lambda,),
    )
    h_list_mev, Deltas_mev, plot_tuples_mev = read_residuals_delta_database_mev(query)
    min_h_mev = min(h_list_mev)
    max_h_mev = max(h_list_mev)

    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_subplot()

    plot_series_cmap(ax, ax.plot, plot_tuples_mev, h_list_mev, min_h_mev, max_h_mev)
    Delta_plot_tuples = [([Delta], [0]) for Delta in Deltas_mev]
    scatter_fn = lambda x, y, **kwargs: ax.scatter(x, y, marker="x", **kwargs)
    sc = plot_series_cmap(
        ax, scatter_fn, Delta_plot_tuples, h_list_mev, min_h_mev, max_h_mev
    )

    cbar = fig.colorbar(sc)
    cbar.set_label("h (meV)")
    ax.scatter([], [], color="black", marker="x", label="Root")
    ax.axhline(y=0, color="gray", linestyle="--")
    ax.set_xlabel(r"$\Delta_0$ (meV)")
    ax.set_ylabel(r"$\delta \Delta$ (meV)")
    ax.set_title(rf"$\lambda = {_lambda}$")
    ax.legend()
    return fig


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
    ax.set_ylabel("Residual (meV)")
    ax.set_title(rf"$\lambda = {_lambda}$")
    ax.legend()
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

    sc = plot_series_cmap_log_scale(
        ax, ax.scatter, plot_tuples, temperatures, min(temperatures), max(temperatures)
    )
    ax.plot(df["temperature"], df["jc"], color="k", zorder=-1)

    ax.set_xlabel(r"Temperature $(K)$")
    ax.set_ylabel(r"$j_c$ $(M A\ m^{-2})$")
    ax.set_xlim(0, 12)
    ax.set_ylim(0.1, None)

    return sc


def get_critical_phase_subplot(ax: Axes, layers_str: str, tunneling: float):
    df = get_critical_current_data(layers_str, tunneling)

    ax.plot(df["temperature"], df["phase"], color="k")

    ax.set_xlabel(r"Temperature $(K)$")
    ax.set_ylabel(r"$\phi_c$ (rad)")
    ax.set_xlim(0, 12)
    y_ticks = [0, np.pi / 2, np.pi]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$"])


def plot_critical_current(layers_str: str, tunneling: float):
    fig = plt.figure(figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 3 * 0.8))

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313, sharex=ax2)

    join_axes_with_shared_x(ax2, ax3)

    pos1 = ax1.get_position()
    ax1.set_position([pos1.x0, pos1.y0 + 0.04, pos1.width, pos1.height])
    plt.setp(ax2.get_xticklabels(), visible=False)

    # Add labels to the plots
    ax1.text(0.05, 0.95, "(a)", transform=ax1.transAxes, va="top", ha="center")
    ax2.text(0.95, 0.95, "(b)", transform=ax2.transAxes, va="top", ha="center")
    ax3.text(0.05, 0.95, "(c)", transform=ax3.transAxes, va="top", ha="center")

    get_current_angle_subplot(ax1, layers_str, tunneling)
    sc = get_critical_current_subplot(ax2, layers_str, tunneling)
    get_critical_phase_subplot(ax3, layers_str, tunneling)

    ax2.set_xlabel("")

    axes = [ax1, ax2, ax3]
    cbar = fig.colorbar(sc, ax=axes)
    cbar.set_label(r"Temperature $(K)$")
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

    ax2.set_xlabel("Temperature (K)")
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
        df = df[df["temperature"] == df["temperature"].min()]
        ax3.plot(
            df["phase"], df["jc"] / df["jc"].max(), label=f"$t = {t * 1e3:.2f}$ meV"
        )
    ax3.set_xlabel(r"$\phi$ (rad)")
    x_ticks = [-np.pi, 0, np.pi]
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    ax3.set_ylabel(r"$j / j_0$")
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

    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    pos4 = ax4.get_position()
    gap_at_bottom = pos2.y0 - pos4.y0
    ax1.set_position(
        [
            pos1.x0,
            pos1.y0 - gap_at_bottom / 2,
            pos1.width,
            pos1.height + gap_at_bottom / 2,
        ]
    )
    ax2.set_position(
        [pos2.x0, pos2.y0 - gap_at_bottom, pos2.width, pos2.height + gap_at_bottom / 2]
    )

    # Add labels to the plots
    ax1.text(0.10, 0.95, "(a)", transform=ax1.transAxes, va="top", ha="center")
    ax2.text(0.90, 0.95, "(b)", transform=ax2.transAxes, va="top", ha="center")
    ax3.text(0.10, 0.95, "(c)", transform=ax3.transAxes, va="top", ha="center")
    ax4.text(0.10, 0.95, "(d)", transform=ax4.transAxes, va="top", ha="center")

    cbar = plt.colorbar(sc, ax=[ax1, ax2, ax3, ax4])
    cbar.set_label(r"$t$ (meV)")
    return fig
