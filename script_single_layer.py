import sqlite3
from copy import deepcopy
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from GKTH.Global_Parameter import GlobalParams
from GKTH.Layer import Layer
from GKTH.self_consistency_delta import (
    GKTH_self_consistency_1S_find_root,
    GKTH_self_consistency_1S_iterate,
    GKTH_self_consistency_1S_residual,
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


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


def get_single_layer_list(_lambda: float):
    layer = Layer(_lambda=_lambda, symmetry="s")
    layer.Delta_0 = 0.01
    layer.tNN = -0.1523
    layer.tNNN = 0
    layer.mu = 0.1025
    return [layer]


def get_single_layer_parameters(h: float):
    return GlobalParams(h=h, a=3.30e-10, nkpoints=300)


def run_residuals_phase(_lambda: float, h_end: float, max_Delta: float, N: int):
    def func(Delta, h):
        # Connect to database
        conn = sqlite3.connect(DATA_DIR / "residuals.db")
        c = conn.cursor()

        # Create table if it doesn't exist
        conn.execute(
            "CREATE TABLE IF NOT EXISTS residuals (lambda REAL, Delta REAL, h REAL, residual REAL)"
        )

        # If data is in the database, skip it
        c.execute(
            f"SELECT * FROM residuals WHERE lambda={_lambda} AND Delta={Delta} AND h={h}"
        )
        query = c.fetchone()
        if query is not None:
            residual = query[3]
            print(f"Delta: {Delta}, h: {h}, residual: {residual}")
            return

        # Get result
        print(
            f"Starting run_residuals_phase, lambda: {_lambda}, Delta: {Delta}, h: {h}"
        )
        p = get_single_layer_parameters(h)
        layers = get_single_layer_list(_lambda)

        residual = GKTH_self_consistency_1S_residual(
            Delta_0_fit=Delta, p=p, layers=layers, layers_to_check=[0]
        )

        # Insert into database
        c.execute(f"INSERT INTO residuals VALUES ({_lambda}, {Delta}, {h}, {residual})")
        conn.commit()
        conn.close()

        print(
            f"Finished run_residuals_phase, lambda: {_lambda}, Delta: {Delta}, h: {h}, residual: {residual}"
        )

    Delta_lin = np.round(np.linspace(0.00, max_Delta, N), 9)
    h_lin = np.round(np.linspace(0.00, h_end, N), 9)

    Delta_mesh, h_mesh = np.meshgrid(Delta_lin, h_lin)

    Parallel(n_jobs=-1)(
        delayed(func)(Delta, h)
        for Delta, h in zip(Delta_mesh.flatten(), h_mesh.flatten())
    )


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


def get_residuals_phase(_lambda: float, max_Delta: float, h_end: float, N: int):
    Delta_lin = np.round(np.linspace(0.00, max_Delta, N), 9)
    h_lin = np.round(np.linspace(0.00, h_end, N), 9)
    Delta_mesh, h_mesh = np.meshgrid(Delta_lin, h_lin)

    # Query residuals from database
    Delta_list = Delta_mesh.flatten()
    h_list = h_mesh.flatten()
    residual_list = get_residuals(_lambda, Delta_list, h_list)
    residual_mesh = np.array(residual_list).reshape(Delta_mesh.shape)

    Delta_mesh_mev = Delta_mesh * 1e3
    h_mesh_mev = h_mesh * 1e3
    residual_mesh_mev = residual_mesh * 1e3

    return Delta_mesh_mev, h_mesh_mev, residual_mesh_mev


def run_for_lambda(_lambda, h_end=1e-3, delta_end=2e-3):
    # Round to take care of floating point errors
    h_list = np.round(np.linspace(0, h_end, 21), 9)
    plot_tuples = []
    Deltas = []

    for h in h_list:
        p = get_single_layer_parameters(h)

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

        print(f"Calculating for lambda: {_lambda}, h: {h}")
        layers = get_single_layer_list(_lambda)

        # See how residuals vary with Delta to check find root
        x_vals, residuals = GKTH_self_consistency_1S_iterate(
            p, deepcopy(layers), max_Delta=delta_end
        )
        plot_tuples.append((x_vals, residuals))

        # Find root where residuals are zero
        Delta, layers = GKTH_self_consistency_1S_find_root(
            p, deepcopy(layers), max_Delta=delta_end
        )
        Deltas.append(Delta)

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


if __name__ == "__main__":
    lambda_list = np.round(np.linspace(0.0, 0.1, 6), 9)
    h_end_list = np.round(np.repeat(1e-3, len(lambda_list)), 9)
    max_Delta_list = np.round(np.repeat(2e-3, len(lambda_list)), 9)

    lambda_list = np.append(lambda_list, [0.1, 0.15, 0.2])
    h_end_list = np.append(h_end_list, [1e-3, 2e-2, 5e-2])
    max_Delta_list = np.append(max_Delta_list, [2e-3, 2e-2, 50e-3])

    def lambda_fn(i):
        run_for_lambda(lambda_list[i], h_end=h_end_list[i], delta_end=max_Delta_list[i])

    # result = Parallel(n_jobs=-1)(delayed(lambda_fn)(i) for i in range(len(lambda_list)))

    tuple_list = [
        (0.1, 1e-3, 2e-3),
        (0.11, 5e-3, 5e-3),
        (0.13, 1e-2, 1e-2),
        (0.15, 2e-2, 2e-2),
        (0.20, 5e-2, 5e-2),
    ]
    # lambda_list = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
    # h_end_list = [1e-3, 5e-3, 5e-3, 1e-2, 2e-2, 2e-2, 3e-2, 3e-2, 4e-2, 4e-2, 5e-2]
    # max_Delta_list = [2e-3, 5e-3, 5e-3, 1e-2, 2e-2, 50e-3]
    N = 41

    for _lambda, h_end, max_Delta in tuple_list:
        run_residuals_phase(_lambda, h_end=h_end, max_Delta=max_Delta, N=N)
