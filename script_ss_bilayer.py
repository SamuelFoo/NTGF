# Script for investigating properties of superconducting bilayers.
# This script examines the dependence of the superconducting gap Δ(T)
# in different superconducting bilayers under various tunneling parameters:
# - Weak s-wave / Strong s-wave
# - Weak d-wave / Strong d-wave
# - Weak s-wave / Strong d-wave
#
# Additionally, the script checks whether there is a difference
# between parabolic and tight-binding dispersions (there isn't).

import pickle
import sqlite3
import traceback
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
from joblib import Parallel, delayed

from GKTH.constants import kB
from GKTH.critical_current import GKTH_critical_current
from GKTH.Global_Parameter import GlobalParams
from GKTH.Green_Function import GKTH_fix_lambda
from GKTH.Greens_current_radial import GKTH_Greens_current_radial
from GKTH.Layer import Layer
from GKTH.self_consistency_delta import GKTH_self_consistency_2S_taketurns

# Global parameters
p = GlobalParams()
p.ntest = 1000
p.nfinal = 250
p.abs_tolerance_self_consistency_1S = 1e-6
p.rel_tolerance_Greens = 1e-6


def load_or_compute_layer(
    layer_name: str, Delta_0: float, Delta_target: float, symmetry="s"
):
    layer_path = Path(f"data/ss_bilayer/layers/{layer_name}.pkl")
    if layer_path.exists():
        return pickle.load(open(layer_path, "rb"))
    else:
        layer = Layer(_lambda=0.0)
        layer.symmetry = symmetry
        layer.Delta_0 = Delta_0
        _lambda = GKTH_fix_lambda(deepcopy(p), deepcopy(layer), Delta_target)
        layer._lambda = _lambda
        pickle.dump(layer, open(layer_path, "wb"))
        return layer


# s-wave high Tc
S1 = load_or_compute_layer("S1", 0.0016, kB * 1.764 * 10)

# s-wave low Tc
S2 = load_or_compute_layer("S2", 0.00083, kB * 1.764 * 5)

# d-wave high Tc
D1 = load_or_compute_layer("D1", 0.0022, kB * 1.764 * 10 * 1.32, symmetry="d")

# d-wave low Tc
D2 = load_or_compute_layer("D2", 0.0012, kB * 1.764 * 5 * 1.32, symmetry="d")

# Define variables
nTs = 51  # Number of temperature points
Ts = np.round(np.linspace(0.0, 0.001, nTs), 9)  # Temperature range
Ts = Ts[1:]  # Remove T=0
nTs = len(Ts)
ts = np.round(np.array([0, 0.25, 0.5, 1, 2, 5, 10]) * 1e-3, 9)  # Tunneling parameters
nts = len(ts)  # Number of tunneling parameters
Deltas = np.zeros((2, nts, nTs))  # Initialize array for storing results

# s-wave tight-binding dispersion
p.nradials = 120


def compute_self_consistency(
    db_path: Path, layers: List[Layer], i: int, lattice_symmetry: str
):
    i1, i2 = np.unravel_index(i, (nts, nTs))
    p1 = deepcopy(p)
    p1.lattice_symmetry = lattice_symmetry
    p1.ts = np.array([ts[i1]])
    p1.T = Ts[i2]

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create table if it doesn't exist
    c.execute(
        """CREATE TABLE IF NOT EXISTS ss_bilayer
                (temperature REAL, tunneling REAL, Ds_0 REAL, Ds_1 REAL)"""
    )

    # If already in database, skip
    c.execute(
        "SELECT Ds_0, Ds_1 FROM ss_bilayer WHERE temperature = ? AND tunneling = ?",
        (p1.T, p1.ts[0]),
    )
    row = c.fetchone()
    if row:
        return row

    print(f"Starting ss: temperature = {p1.T}, tunneling = {p1.ts}, {i} of {nts * nTs}")
    try:
        Ds, _, _ = GKTH_self_consistency_2S_taketurns(p1, layers)
        # Save results to sql
        conn.execute(
            "INSERT INTO ss_bilayer (temperature, tunneling, Ds_0, Ds_1) VALUES (?, ?, ?, ?)",
            (p1.T, p1.ts[0], Ds[0], Ds[1]),
        )
        conn.commit()
        conn.close()
        print(
            f"Finished ss: temperature = {p1.T}, tunneling = {p1.ts}, {i} of {nts * nTs}"
        )
        return Ds, i

    except Exception as e:
        print(f"Error: {e}")
        conn.close()
        return None


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
        # Save results to sql
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

        # Save results to sql
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
        # Save results to sql
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


# S-S
# ss_fn = lambda i: compute_self_consistency(
#     Path("data/ss_bilayer/ss_bilayer.db"), [S1, S2], i, "mm"
# )
# results = Parallel(n_jobs=-1)(delayed(ss_fn)(i) for i in range(nts * nTs))

# D-D
# dd_fn = lambda i: compute_self_consistency(
#     Path("data/ss_bilayer/dd_bilayer.db"), [D1, D2], i, "4mm"
# )
# results = Parallel(n_jobs=-1)(delayed(dd_fn)(i) for i in range(nts * nTs))

# Big D-Small S
# ds_fn = lambda i: compute_self_consistency(
#     Path("data/ss_bilayer/ds_bilayer.db"), [D1, S2], i, "mm"
# )
# results = Parallel(n_jobs=-1)(delayed(ds_fn)(i) for i in range(nts * nTs))

# Big S-Small D
# sd_fn = lambda i: compute_self_consistency(
#     Path("data/ss_bilayer/sd_bilayer.db"), [S1, D2], i, "mm"
# )
# results = Parallel(n_jobs=-1)(delayed(sd_fn)(i) for i in range(nts * nTs))

# Compute critical current
t = np.round(0.5e-3, 9)  # Tunneling parameter

# Critical currents
# nTs = 51  # Number of temperature points
# Ts = np.round(np.linspace(0.0, 0.001, nTs), 9)  # Temperature range
# Ts = Ts[1:]  # Remove T=0

# SNS
# N = Layer(_lambda=0.0, symmetry="n")
# sns_fn = lambda T: compute_critical_current("S1_N_S2", [S1, N, S2], "mm", t=t, T=T)
# results = Parallel(n_jobs=-1)(delayed(sns_fn)(T) for T in Ts)

# sns_fn = lambda T: compute_critical_current("S1_N_S1", [S1, N, S1], "mm", t=t, T=T)
# results = Parallel(n_jobs=-1)(delayed(sns_fn)(T) for T in Ts)

# sns_fn = lambda T: compute_critical_current("S2_N_S2", [S2, N, S2], "mm", t=t, T=T)
# results = Parallel(n_jobs=-1)(delayed(sns_fn)(T) for T in Ts)

# Current vs Phase
# nTs = 21
# Ts = np.round(np.linspace(0.0, 0.001, nTs), 9)  # Temperature range
# Ts = Ts[1:]  # Remove T=0
# phases = np.round(np.linspace(-np.pi, np.pi, 21), 9)
# T_mesh, phase_mesh = np.meshgrid(Ts, phases)
# T_list = T_mesh.flatten()
# phase_list = phase_mesh.flatten()

# sns_fn = lambda i: compute_current(
#     "S1_N_S2", [S1, N, S2], "mm", t=t, T=T_list[i], phase=phase_list[i]
# )
# results = Parallel(n_jobs=-1)(delayed(sns_fn)(i) for i in range(len(T_list)))

# sns_fn = lambda i: compute_current(
#     "S1_N_S1", [S1, N, S1], "mm", t=t, T=T_list[i], phase=phase_list[i]
# )
# results = Parallel(n_jobs=-1)(delayed(sns_fn)(i) for i in range(len(T_list)))

# sns_fn = lambda i: compute_current(
#     "S2_N_S2", [S2, N, S2], "mm", t=t, T=T_list[i], phase=phase_list[i]
# )
# results = Parallel(n_jobs=-1)(delayed(sns_fn)(i) for i in range(len(T_list)))

# SFS
S1_hs = deepcopy(S1)
S1_hs.h = 0.5e-3
S2_hs = deepcopy(S2)
S2_hs.h = 0.5e-3

F = Layer(_lambda=0.0, symmetry="n")
F.h = 10e-3
phases = np.round(np.linspace(-np.pi, np.pi, 21), 9)

T = 1 * kB

dE = 1e3
sfs_fn = lambda phase: compute_critical_current_ferromagnet(
    "S1_F_S2", [S1_hs, F, S2_hs], "mm", t=t, T=T, phase=phase, dE=dE
)
results = Parallel(n_jobs=-1)(delayed(sfs_fn)(phase) for phase in phases)
