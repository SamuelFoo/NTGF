import pickle
from copy import deepcopy
from pathlib import Path

from GKTH.constants import kB
from GKTH.Global_Parameter import GlobalParams
from GKTH.Green_Function import GKTH_fix_lambda
from GKTH.Layer import Layer

# Global parameters
p = GlobalParams()
p.ntest = 1000
p.nfinal = 250
p.abs_tolerance_self_consistency_1S = 1e-6
p.rel_tolerance_Greens = 1e-6
p.nradials = 120


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
