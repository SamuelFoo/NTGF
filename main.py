import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Global_Parameter import GlobalParams
from Layer import Layer
from self_consistency_delta import (
    GKTH_self_consistency_1S_find_root,
    GKTH_self_consistency_1S_iterate,
)

DATA_DIR =  Path("data")
DATA_DIR.mkdir(exist_ok=True)

PRESENTATION_MEDIA_DIR = Path("presentation_media")
PRESENTATION_MEDIA_DIR.mkdir(exist_ok=True)

def run_for_lambda(_lambda):
    # Round to take care of floating point errors
    h_list = np.round(np.linspace(0, 1e-3, 21), 9)
    plot_tuples = []
    Deltas = []

    for h in h_list:
        p = GlobalParams(h=h)

        # Connect to database
        conn = sqlite3.connect(DATA_DIR / "residual_delta.db")
        c = conn.cursor()

        # Create table if it doesn't exist
        c.execute(
            """CREATE TABLE IF NOT EXISTS results
                    (_lambda REAL, h REAL, Delta REAL, layers STRING, x_vals STRING, residuals STRING)"""
        )

        # If already in database, skip
        c.execute(
            "SELECT Delta FROM results WHERE _lambda = ? AND h = ?",
            (_lambda, h),
        )
        if c.fetchone():
            continue

        # Find root where residuals are zero
        layers = [Layer(_lambda=_lambda)]
        Delta, layers = GKTH_self_consistency_1S_find_root(p, layers)
        Deltas.append(Delta)

        # See how residuals vary with Delta to check find root
        layers = [Layer(_lambda=_lambda)]
        x_vals, residuals = GKTH_self_consistency_1S_iterate(p, layers)
        plot_tuples.append((x_vals, residuals))

        # Insert the data
        layers_data = list(map(lambda l: l.__dict__, layers))
        c.execute(
            "INSERT INTO results (_lambda, h, Delta, layers, x_vals, residuals) VALUES (?, ?, ?, ?, ?, ?)",
            (
                _lambda,
                h,
                Delta,
                repr(layers_data),
                repr(x_vals.tolist()),
                repr(residuals),
            ),
        ),

        # Commit the changes and close the connection
        conn.commit()
        conn.close()

def plot_for_lambda(_lambda):
    # Connect to the database
    conn = sqlite3.connect(DATA_DIR / "residual_delta.db")
    c = conn.cursor()

    # Read the data from the database
    c.execute(
        "SELECT h, Delta, x_vals, residuals FROM results WHERE _lambda = ?", (_lambda,)
    )
    rows = c.fetchall()

    # Close the connection
    conn.close()

    # Process the data
    plot_tuples = []
    Deltas = []
    h_list = []

    for row in rows:
        h, Delta, x_vals_str, residuals_str = row
        h_list.append(h)
        Deltas.append(Delta)
        x_vals = eval(x_vals_str)
        residuals = eval(residuals_str)
        plot_tuples.append((x_vals, residuals))

    plt.figure(figsize=(12, 6))

    sc = plt.scatter([], [], c=[], cmap="copper", vmin=min(h_list), vmax=max(h_list))
    cbar = plt.colorbar(sc)
    cbar.set_label("h (eV)")

    for i, (h, (x_vals, residuals), Delta) in enumerate(zip(h_list, plot_tuples, Deltas)):
        color = plt.cm.copper((len(h_list) - i) / len(h_list))
        plt.plot(x_vals, residuals, color=color)
        plt.scatter([Delta], [0], color=color, marker="x")

    plt.scatter([], [], color="black", marker="x", label="Root")

    plt.axhline(y=0, color="gray", linestyle="--")
    plt.xlabel(r"$\Delta_0$ (eV)")
    plt.ylabel("Residual (eV)")
    plt.title(rf"$\lambda = {_lambda}$")
    plt.legend()
    plt.savefig(
        PRESENTATION_MEDIA_DIR / f"residuals_delta_lambda_{_lambda}.svg", transparent=True
    )
    plt.show()