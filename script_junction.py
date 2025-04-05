import sqlite3
import traceback
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
from joblib import Parallel, delayed

from GKTH.critical_current import GKTH_critical_current
from GKTH.Global_Parameter import GlobalParams
from GKTH.Greens_current_radial import GKTH_Greens_current_radial
from GKTH.Layer import Layer
from script_layer import S1, S2

# Global parameters
p = GlobalParams()
p.ntest = 1000
p.nfinal = 250
p.abs_tolerance_self_consistency_1S = 1e-6
p.rel_tolerance_Greens = 1e-6
p.nradials = 120

# Define variables
nTs = 51  # Number of temperature points
Ts = np.round(np.linspace(0.0, 0.001, nTs), 9)  # Temperature range
Ts = Ts[1:]  # Remove T=0
nTs = len(Ts)
ts = np.round(np.array([0, 0.25, 0.5, 1, 2, 5, 10]) * 1e-3, 9)  # Tunneling parameters
nts = len(ts)  # Number of tunneling parameters
Deltas = np.zeros((2, nts, nTs))  # Initialize array for storing results


def compute_critical_current(
    db_name: str, layers: List[Layer], lattice_symmetry: str, t: float, T: float
):
    p1 = deepcopy(p)
    p1.lattice_symmetry = lattice_symmetry
    p1.ts = np.array([t, t])
    p1.T = T

    db_path = Path("data/critical_current") / f"{db_name}_critical.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create table if it doesn't exist
    c.execute(
        """CREATE TABLE IF NOT EXISTS current
                (temperature REAL, tunneling REAL, jc REAL, phase REAL)"""
    )

    # If already in database, skip
    c.execute(
        "SELECT jc, phase FROM current WHERE temperature = ? AND tunneling = ?",
        (p1.T, p1.ts[0]),
    )
    row = c.fetchone()
    if row:
        return row

    print(f"Starting critical current: temperature = {p1.T}, tunneling = {p1.ts}")
    try:
        jc, phase = GKTH_critical_current(p1, layers, db_name=db_name)
        conn.execute(
            "INSERT INTO current (temperature, tunneling, jc, phase) VALUES (?, ?, ?, ?)",
            (p1.T, p1.ts[0], jc, phase),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error computing critical current: {e}")
        traceback.print_exc()
    print(f"Finished critical current: temperature = {p1.T}, tunneling = {p1.ts}")


def compute_critical_current_ferromagnet(
    db_name: str,
    layers: List[Layer],
    lattice_symmetry: str,
    t: float,
    T: float,
    phase: float,
    dE: float,
):
    p1 = deepcopy(p)
    p1.lattice_symmetry = lattice_symmetry
    p1.ts = np.array([t, t])
    p1.T = T

    db_path = Path("data/critical_current") / f"{db_name}_critical.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create table if it doesn't exist
    c.execute(
        """CREATE TABLE IF NOT EXISTS current
                (temperature REAL, tunneling REAL, jc REAL, phase REAL, dE REAL)"""
    )

    # If already in database, skip
    c.execute(
        "SELECT jc, phase FROM current WHERE temperature = ? AND tunneling = ? AND phase = ? AND dE = ?",
        (p1.T, p1.ts[0], phase, dE),
    )
    row = c.fetchone()
    if row:
        return row

    print(
        f"Starting current: temperature = {p1.T}, tunneling = {p1.ts}, phase = {phase}, dE = {dE}"
    )
    try:
        layers_temp = deepcopy(layers)
        layers_temp[0].theta_ip = phase
        layers_temp[2].theta_ip = phase
        layers_temp[1].dE = dE
        jc, phase = GKTH_critical_current(
            p1, layers, db_name=db_name, spin_current=True
        )
        print("\n\n\n")
        print(jc)
        print("\n\n\n")

        conn.execute(
            "INSERT INTO current (temperature, tunneling, jc, phase, dE) VALUES (?, ?, ?, ?, ?)",
            (p1.T, p1.ts[0], jc[0, 0], phase, dE),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error computing current: {e}")
        traceback.print_exc()
    print(
        f"Finished current: temperature = {p1.T}, tunneling = {p1.ts}, phase = {phase}, dE = {dE}"
    )


def compute_current(
    db_name: str,
    layers: List[Layer],
    lattice_symmetry: str,
    t: float,
    T: float,
    phase: float,
):
    p1 = deepcopy(p)
    p1.lattice_symmetry = lattice_symmetry
    p1.ts = np.array([t, t])
    p1.T = T

    db_path = Path("data/current") / f"{db_name}_current.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create table if it doesn't exist
    c.execute(
        """CREATE TABLE IF NOT EXISTS current
                (temperature REAL, tunneling REAL, jc REAL, phase REAL)"""
    )

    # If already in database, skip
    c.execute(
        "SELECT jc, phase FROM current WHERE temperature = ? AND tunneling = ? AND phase = ?",
        (p1.T, p1.ts[0], phase),
    )
    row = c.fetchone()
    if row:
        return row

    print(
        f"Starting current: temperature = {p1.T}, tunneling = {p1.ts}, phase = {phase}"
    )
    try:
        layers_temp = deepcopy(layers)
        layers_temp[2].phi = phase
        j_t, _, _, _, _, _ = GKTH_Greens_current_radial(
            p1, layers_temp, db_name=db_name
        )
        conn.execute(
            "INSERT INTO current (temperature, tunneling, jc, phase) VALUES (?, ?, ?, ?)",
            (p1.T, p1.ts[0], j_t[0, 0], phase),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error computing current: {e}")
        traceback.print_exc()
    print(
        f"Finished current: temperature = {p1.T}, tunneling = {p1.ts}, phase = {phase}"
    )


if __name__ == "__main__":
    # SNS
    # Compute critical current
    tunneling_params = np.array([0.1, 0.25, 0.5, 1, 2, 5, 10]) * 1e-3
    tunneling_params = np.round(tunneling_params, 9)

    nTs = 51  # Number of temperature points
    Ts = np.round(np.linspace(0.0, 0.001, nTs), 9)  # Temperature range
    Ts = Ts[1:]  # Remove T=0

    for t in tunneling_params:
        N = Layer(_lambda=0.0, symmetry="n")
        sns_fn = lambda T: compute_critical_current(
            "S1_N_S2", [S1, N, S2], "mm", t=t, T=T
        )
        results = Parallel(n_jobs=-1)(delayed(sns_fn)(T) for T in Ts)

    t = 0.5e-3
    sns_fn = lambda T: compute_critical_current("S1_N_S1", [S1, N, S1], "mm", t=t, T=T)
    results = Parallel(n_jobs=-1)(delayed(sns_fn)(T) for T in Ts)

    sns_fn = lambda T: compute_critical_current("S2_N_S2", [S2, N, S2], "mm", t=t, T=T)
    results = Parallel(n_jobs=-1)(delayed(sns_fn)(T) for T in Ts)

    # Current vs Phase
    nTs = 21
    Ts = np.round(np.linspace(0.0, 0.001, nTs), 9)  # Temperature range
    Ts = Ts[1:]  # Remove T=0
    phases = np.round(np.linspace(-np.pi, np.pi, 21), 9)
    T_mesh, phase_mesh = np.meshgrid(Ts, phases)
    T_list = T_mesh.flatten()
    phase_list = phase_mesh.flatten()

    for t in tunneling_params:
        sns_fn = lambda i: compute_current(
            "S1_N_S2", [S1, N, S2], "mm", t=t, T=T_list[i], phase=phase_list[i]
        )
        results = Parallel(n_jobs=-1)(delayed(sns_fn)(i) for i in range(len(T_list)))

    t = 0.5e-3
    sns_fn = lambda i: compute_current(
        "S1_N_S1", [S1, N, S1], "mm", t=t, T=T_list[i], phase=phase_list[i]
    )
    results = Parallel(n_jobs=-1)(delayed(sns_fn)(i) for i in range(len(T_list)))

    sns_fn = lambda i: compute_current(
        "S2_N_S2", [S2, N, S2], "mm", t=t, T=T_list[i], phase=phase_list[i]
    )
    results = Parallel(n_jobs=-1)(delayed(sns_fn)(i) for i in range(len(T_list)))

    # SFS
    # S1_hs = deepcopy(S1)
    # S1_hs.h = 0.5e-3
    # S2_hs = deepcopy(S2)
    # S2_hs.h = 0.5e-3

    # F = Layer(_lambda=0.0, symmetry="n")
    # F.h = 10e-3
    # phases = np.round(np.linspace(-np.pi, np.pi, 21), 9)

    # T = 1 * kB
    # t = np.round(0.5e-3, 9)  # Tunneling parameter

    # dE = 1e3
    # sfs_fn = lambda phase: compute_critical_current_ferromagnet(
    #     "S1_F_S2", [S1_hs, F, S2_hs], "mm", t=t, T=T, phase=phase, dE=dE
    # )
    # results = Parallel(n_jobs=-1)(delayed(sfs_fn)(phase) for phase in phases)
