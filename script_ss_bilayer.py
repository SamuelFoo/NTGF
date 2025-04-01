# Script for investigating properties of superconducting bilayers.
# This script examines the dependence of the superconducting gap Δ(T)
# in different superconducting bilayers under various tunneling parameters:
# - Weak s-wave / Strong s-wave
# - Weak d-wave / Strong d-wave
# - Weak s-wave / Strong d-wave
#
# Additionally, the script checks whether there is a difference
# between parabolic and tight-binding dispersions (there isn't).

import copy
import pickle
import sqlite3
from pathlib import Path
from typing import List

import numpy as np
from joblib import Parallel, delayed

from constants import kB
from Global_Parameter import GlobalParams
from Green_Function import GKTH_fix_lambda
from Layer import Layer
from self_consistency_delta import GKTH_self_consistency_2S_taketurns

# Global parameters
p = GlobalParams()
p.ntest = 1000
p.nfinal = 250
p.abs_tolerance_self_consistency_1S = 1e-6
p.rel_tolerance_Greens = 1e-6


def load_or_compute_layer(layer_name, Delta_0, Delta_target, symmetry="s"):
    layer_path = Path(f"data/ss_bilayer/layers/{layer_name}.pkl")
    if layer_path.exists():
        return pickle.load(open(layer_path, "rb"))
    else:
        layer = Layer(_lambda=0.0)
        layer.symmetry = symmetry
        layer.Delta_0 = Delta_0
        _, layer = GKTH_fix_lambda(p, layer, Delta_target)
        pickle.dump(layer, open(layer_path, "wb"))
        return layer


# s-wave high Tc
S1 = load_or_compute_layer("S1", 0.0016, kB * 1.764 * 10)

# s-wave low Tc
S2 = load_or_compute_layer("S2", 0.00083, kB * 1.764 * 5)

# d-wave high Tc
# print(kB * 1.764 * 10 * 1.32)
D1 = load_or_compute_layer("D1", 0.0022, kB * 1.764 * 10 * 1.32, symmetry="d")
# %D1.lambda=GKTH_fix_lambda(p,D1,0.0023512)

# d-wave low Tc
# print(kB * 1.764 * 5 * 1.32)
D2 = load_or_compute_layer("D2", 0.0012, kB * 1.764 * 5 * 1.32, symmetry="d")
# %D2.lambda=GKTH_fix_lambda(p,D2,0.0010273);

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
    p1 = copy.deepcopy(p)
    p1.lattice_symmetry = lattice_symmetry
    p1.ts = [ts[i1]]
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
    Ds, _, _ = GKTH_self_consistency_2S_taketurns(p1, layers)
    print(f"Finished ss: temperature = {p1.T}, tunneling = {p1.ts}, {i} of {nts * nTs}")

    # Save results to sql
    conn.execute(
        "INSERT INTO ss_bilayer (temperature, tunneling, Ds_0, Ds_1) VALUES (?, ?, ?, ?)",
        (p1.T, p1.ts[0], Ds[0], Ds[1]),
    )
    conn.commit()
    conn.close()
    return Ds, i


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
sd_fn = lambda i: compute_self_consistency(
    Path("data/ss_bilayer/sd_bilayer.db"), [S1, D2], i, "mm"
)
results = Parallel(n_jobs=-1)(delayed(sd_fn)(i) for i in range(nts * nTs))
