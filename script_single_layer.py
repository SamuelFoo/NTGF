import sqlite3
from pathlib import Path

import numpy as np
from skimage import measure

from GKTH.Global_Parameter import GlobalParams
from GKTH.Layer import Layer
from GKTH.self_consistency_delta import (
    GKTH_self_consistency_1S_find_root,
    GKTH_self_consistency_1S_iterate,
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

PRESENTATION_MEDIA_DIR = Path("presentation_media")
PRESENTATION_MEDIA_DIR.mkdir(exist_ok=True)


def read_residual_delta_database(query: tuple):
    # Connect to the database
    conn = sqlite3.connect(DATA_DIR / "residual_delta.db")
    c = conn.cursor()

    # Read the data from the database
    c.execute(*query)
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

    h_list = np.array(h_list)
    Deltas = np.array(Deltas)

    return h_list, Deltas, plot_tuples


def read_residuals_delta_database_mev(query: tuple):
    h_list, Deltas, plot_tuples = read_residual_delta_database(query)

    h_list_mev = h_list * 1e3
    Deltas_mev = Deltas * 1e3
    plot_tuples_mev = [
        (np.array(x_vals) * 1e3, np.array(residuals) * 1e3)
        for x_vals, residuals in plot_tuples
    ]
    return h_list_mev, Deltas_mev, plot_tuples_mev


def get_delta_vs_h(_lambda):
    query = (
        "SELECT h, Delta, x_vals, residuals FROM results WHERE _lambda = ?",
        (_lambda,),
    )
    h_list, Deltas, _ = read_residual_delta_database(query)
    return h_list, Deltas


def get_residuals(_lambda, Delta_list, h_list):
    residual_list = []

    for Delta, h in zip(Delta_list, h_list):
        conn = sqlite3.connect(DATA_DIR / "residuals.db")
        c = conn.cursor()

        c.execute(
            f"SELECT * FROM residuals WHERE lambda={_lambda} AND Delta={Delta} AND h={h}"
        )
        query = c.fetchone()
        residual = query[3]
        residual_list.append(residual)

        conn.close()

    return residual_list


def run_for_lambda(_lambda, h_end=1e-3, delta_end=2e-3):
    # Round to take care of floating point errors
    h_list = np.round(np.linspace(0, h_end, 21), 9)
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
        Delta, layers = GKTH_self_consistency_1S_find_root(
            p, layers, max_Delta=delta_end
        )
        Deltas.append(Delta)

        # See how residuals vary with Delta to check find root
        layers = [Layer(_lambda=_lambda)]
        x_vals, residuals = GKTH_self_consistency_1S_iterate(
            p, layers, max_Delta=delta_end
        )
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


def drop_lambda(_lambda):
    conn = sqlite3.connect(DATA_DIR / "residual_delta.db")
    c = conn.cursor()
    c.execute("DELETE FROM results WHERE _lambda = ?", (_lambda,))
    conn.commit()
    conn.close()


def get_contour(
    x_mesh: np.ndarray, y_mesh: np.ndarray, z_mesh: np.ndarray, value: np.float64
):
    contours = measure.find_contours(z_mesh, value)
    contour = contours[0]
    x_scale = 1 / x_mesh.shape[1] * x_mesh.max()
    y_scale = 1 / y_mesh.shape[1] * y_mesh.max()
    return contour[:, 1] * x_scale, contour[:, 0] * y_scale
