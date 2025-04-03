import sqlite3
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm

from GKTH.constants import kB
from script_single_layer import (
    DATA_DIR,
    get_delta_vs_h,
    read_residual_delta_database,
    read_residuals_delta_database_mev,
)

FIGURE_SIZE = (8, 4)


def plot_series_cmap(
    ax: Axes, plot_fn: Callable, series_list, series_cmap_values, cmap_min, cmap_max
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
    ax.set_ylabel("Residual (meV)")
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


def plot_current_angle(db_name):
    conn = sqlite3.connect(DATA_DIR / "current" / db_name)
    query = "SELECT temperature, tunneling, jc, phase FROM current"
    df = pd.read_sql_query(query, conn)

    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_subplot(111)

    plot_tuples = []
    temperatures = []
    for i, temperature in enumerate(df["temperature"].unique()):
        subset = df[df["temperature"] == temperature]
        plot_tuples.append((subset["phase"], subset["jc"] / 1e6))
        temperatures.append(temperature / kB)

    sc = plot_series_cmap_log_scale(
        ax, ax.plot, plot_tuples, temperatures, min(temperatures), max(temperatures)
    )
    cbar = fig.colorbar(sc)
    cbar.set_label("Temperature (K)")
    tick_values = [1.0, 2.0, 4.0, 8.0]
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{t:.1f}" for t in tick_values])

    ax.set_xlabel("Phase (rad)")
    x_ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

    ax.set_ylabel(r"Current Density $(M A\ m^{-2})$")
    ax.set_xlim(-np.pi, np.pi)
    return fig


def plot_critical_current(db_name):
    conn = sqlite3.connect(DATA_DIR / "critical_current" / db_name)
    query = "SELECT temperature, tunneling, jc, phase FROM current"
    df = pd.read_sql_query(query, conn)

    axes: List[Axes]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 2),
        sharex="all",
        gridspec_kw={"hspace": 0},
    )

    plot_tuples = []
    temperatures = []
    for i, temperature in enumerate(df["temperature"].unique()):
        subset = df[df["temperature"] == temperature]
        plot_tuples.append(([temperature / kB], subset["jc"] / 1e6))
        temperatures.append(temperature / kB)

    sc = plot_series_cmap_log_scale(
        axes[0],
        axes[0].scatter,
        plot_tuples,
        temperatures,
        min(temperatures),
        max(temperatures),
    )
    sorted_idxs = np.argsort(df["temperature"])
    sorted_df = df.iloc[sorted_idxs]
    axes[0].plot(
        sorted_df["temperature"] / kB, sorted_df["jc"] / 1e6, color="k", zorder=-1
    )

    axes[0].set_ylabel(r"Critical Current Density $(M A\ m^{-2})$")
    axes[0].set_ylim(0.1, None)

    axes[1].plot(sorted_df["temperature"] / kB, sorted_df["phase"], color="k")
    axes[1].set_xlabel(r"Temperature $(K)$")
    axes[1].set_xlim(0, 12)
    axes[1].set_ylabel("Phase (rad)")
    y_ticks = [0, np.pi / 2, np.pi]
    axes[1].set_yticks(y_ticks)
    axes[1].set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$"])

    cbar = fig.colorbar(sc, ax=axes.ravel().tolist())
    cbar.set_label(r"Temperature $(K)$")
    tick_values = [1.0, 2.0, 4.0, 8.0]
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{t:.1f}" for t in tick_values])

    return fig
