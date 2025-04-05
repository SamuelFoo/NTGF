# Script for investigating properties of superconducting bilayers.
# This script examines the dependence of the superconducting gap Δ(T)
# in different superconducting bilayers under various tunneling parameters:
# - Weak s-wave / Strong s-wave
# - Weak d-wave / Strong d-wave
# - Weak s-wave / Strong d-wave
#
# Additionally, the script checks whether there is a difference
# between parabolic and tight-binding dispersions (there isn't).

import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
from joblib import Parallel, delayed

from GKTH.Global_Parameter import GlobalParams
from GKTH.Layer import Layer
from GKTH.self_consistency_delta import GKTH_self_consistency_2S_taketurns
from script_layer import D1, D2, S1, S2

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


if __name__ == "__main__":
    # S-S
    ss_fn = lambda i: compute_self_consistency(
        Path("data/ss_bilayer/ss_bilayer.db"), [S1, S2], i, "mm"
    )
    results = Parallel(n_jobs=-1)(delayed(ss_fn)(i) for i in range(nts * nTs))

    # D-D
    dd_fn = lambda i: compute_self_consistency(
        Path("data/ss_bilayer/dd_bilayer.db"), [D1, D2], i, "4mm"
    )
    results = Parallel(n_jobs=-1)(delayed(dd_fn)(i) for i in range(nts * nTs))

    # Big D-Small S
    ds_fn = lambda i: compute_self_consistency(
        Path("data/ss_bilayer/ds_bilayer.db"), [D1, S2], i, "mm"
    )
    results = Parallel(n_jobs=-1)(delayed(ds_fn)(i) for i in range(nts * nTs))

    # Big S-Small D
    sd_fn = lambda i: compute_self_consistency(
        Path("data/ss_bilayer/sd_bilayer.db"), [S1, D2], i, "mm"
    )
    results = Parallel(n_jobs=-1)(delayed(sd_fn)(i) for i in range(nts * nTs))
