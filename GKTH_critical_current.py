import sqlite3
import time
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
from scipy.optimize import fmin

from GKTH_Greens_current_radial import GKTH_Greens_current_radial
from Global_Parameter import GlobalParams
from Layer import Layer


def GKTH_critical_current(p: GlobalParams, layers: List[Layer], db_name: str, **kwargs):
    """
    Finds the maximum current vs phase for a given structure

    Parameters:
    -----------
    p : object
        Global Parameter object defining the stack and calculation parameters.
    layers : list
        An array of Layer objects defining the stack

    Optional Parameters:
    -------------------
    layer_to_vary : int
        which layer to change the phase, default last
    initial_guess : float
        initial estimate of critical phase
    maxCalcs : int
        maximum calculations in current calculation
    spin_current : bool
        whether or not to return the spin-currents

    Returns:
    --------
    jc : float or array
        the critical current
    phase : float
        the critical phase
    """
    # Set default values for optional parameters
    layer_to_vary = kwargs.get("layer_to_vary", np.nan)
    initial_guess = kwargs.get("initial_guess", 1.5)
    maxCalcs = kwargs.get("maxCalcs", 500)
    spin_current = kwargs.get("spin_current", False)

    # If no layer to vary set, make it the last layer
    if np.isnan(layer_to_vary):
        layer_to_vary = len(layers) - 1

    # The function to minimize to get jc. Returns -jc so a minimizer
    # can be used to find the maximum jc
    def jc_function(xs):
        start_time = time.time()

        if np.isscalar(xs):
            xs = [xs]
        js = np.zeros(len(xs))

        # TODO: Update saving logic when len(xs) > 1s
        assert len(xs) == 1, "Only one value of x is allowed at a time"
        for i in range(len(xs)):
            x = xs[i]
            x = np.round(x, 9)

            is_present, j = check_database(T=p.T, t=p.ts[0], x=x, db_name=db_name)

            if is_present:
                js[i] = -j
            else:
                layers_temp = layers.copy()
                layers_temp[layer_to_vary].phi = x
                j_t, _, _, _, _, _ = GKTH_Greens_current_radial(
                    p, layers_temp, maxCalcs=maxCalcs, include_spin=spin_current
                )
                print("j_t", j_t)
                js[i] = -j_t[0, 0]
                print("js[i]", js[i])

                save_current(T=p.T, t=p.ts[0], x=x, j=j_t[0, 0], db_name=db_name)

        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        return js[0] if len(js) == 1 else js

    # The minimum search
    phase = fmin(jc_function, initial_guess, xtol=1e-3, ftol=1e10, disp=False)[0]
    phase = phase % (2 * np.pi)

    # Force the result between -pi and pi
    if phase > np.pi:
        phase = phase - 2 * np.pi

    if spin_current:
        layers_copy = deepcopy(layers)
        layers_copy[layer_to_vary].phi = phase
        jc = GKTH_Greens_current_radial(
            deepcopy(p), layers_copy, maxCalcs=maxCalcs, include_spin=True
        )
    else:
        jc = -jc_function(phase)

    return jc, phase


def check_database(T: float, t: float, x: float, db_name: str):
    db_path = Path("data/critical_current") / f"{db_name}_currents.db"

    if not db_path.exists():
        return False, None

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute(
        f"""CREATE TABLE IF NOT EXISTS {db_name}_currents
                (temperature REAL, tunneling REAL, x REAL, j REAL)"""
    )

    c.execute(
        f"SELECT j FROM {db_name}_currents WHERE temperature = ? AND tunneling = ? AND x = ?",
        (T, t, x),
    )
    row = c.fetchone()

    if row:
        j = row[0]
        print(f"Found j for T={T}, t={t}, x={x}: {j}")
        return True, j
    else:
        return False, None


def save_current(T: float, t: float, x: float, j: float, db_name: str):
    db_path = Path("data/critical_current") / f"{db_name}_currents.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute(
        f"""CREATE TABLE IF NOT EXISTS {db_name}_currents
                (temperature REAL, tunneling REAL, x REAL, j REAL)"""
    )

    c.execute(
        f"INSERT INTO {db_name}_currents (temperature, tunneling, x, j) VALUES (?, ?, ?, ?)",
        (T, t, x, j),
    )

    conn.commit()
    conn.close()
