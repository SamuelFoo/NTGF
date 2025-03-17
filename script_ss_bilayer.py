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

# s-wave high Tc
S1 = Layer(_lambda=0.0)
S1.Delta_0 = 0.0016
_, S1 = GKTH_fix_lambda(p, S1, kB * 1.764 * 10)

# s-wave low Tc
S2 = Layer(_lambda=0.0)
S2.Delta_0 = 0.00083
_, S2 = GKTH_fix_lambda(p, S2, kB * 1.764 * 5)

# d-wave high Tc
D1 = Layer(_lambda=0.0)
D1.symmetry = "d"
D1.Delta_0 = 0.0022
# %D1.lambda=GKTH_fix_lambda(p,D1,0.0023512)
_, D1 = GKTH_fix_lambda(p, D1, kB * 1.764 * 10 * 1.32)

# d-wave low Tc
D2 = Layer(_lambda=0.0)
D2.symmetry = "d"
D2.Delta_0 = 0.0012
# %D2.lambda=GKTH_fix_lambda(p,D2,0.0010273);
_, D2 = GKTH_fix_lambda(p, D1, kB * 1.764 * 5 * 1.32)

# Define variables
nTs = 50  # Number of temperature points
Ts = np.linspace(0.00001, 0.001, nTs)  # Temperature range
ts = np.array([0, 0.25, 0.5, 1, 2, 5, 10]) * 1e-3  # Tunneling parameters
nts = len(ts)  # Number of tunneling parameters
Deltas = np.zeros((2, nts, nTs))  # Initialize array for storing results

# s-wave tight-binding dispersion
p.nradials = 120
p.lattice_symmetry = "mm"


def compute_self_consistency(i):
    i1, i2 = np.unravel_index(i, (nts, nTs))
    p1 = copy.deepcopy(p)
    p1.ts = [ts[i1]]
    p1.T = Ts[i2]
    Ds, _, _ = GKTH_self_consistency_2S_taketurns(p1, [S1, S2])
    print(f"ss: {i} of {nts * nTs}")
    return Ds, i


results = Parallel(n_jobs=-1)(
    delayed(compute_self_consistency)(i) for i in range(nts * nTs)
)

# Store results in the Deltas array
for Ds, i in results:
    Deltas[:, i] = Ds

# Save the results to a file
np.savez("data/ss_bilayer/ss_Delta_T_t_tb.npz", Ts=Ts, Deltas=Deltas, ts=ts)
