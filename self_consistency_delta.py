from typing import List

import numpy as np
from scipy.optimize import root_scalar

from Global_Parameter import GlobalParams
from Green_Function import GKTH_Greens
from Layer import Layer


# Residual function for self-consistency
def GKTH_self_consistency_1S_residual(
    Delta_0_fit: float, p: GlobalParams, layers: List[Layer], layers_to_check: List[int]
):
    # Update Delta_0 for layers being checked
    for i in layers_to_check:
        layers[i].Delta_0 = Delta_0_fit

    # Calculate the anomalous Green function sums
    Fs_sums = GKTH_Greens(p, layers)
    idx = layers_to_check[0]
    residual = layers[idx]._lambda * p.T * np.abs(Fs_sums[idx]) - Delta_0_fit
    print(f"delta_0 residual: {Delta_0_fit:.3e} {residual:.4e}")
    return residual


def GKTH_self_consistency_1S_iterate(
    p: GlobalParams,
    layers: List[Layer],
    layers_to_check: list[int] = [0],
):
    x_vals = np.linspace(0, 0.002, 50)
    residuals = []
    for x in x_vals:
        residual = GKTH_self_consistency_1S_residual(
            Delta_0_fit=x, p=p, layers=layers, layers_to_check=layers_to_check
        )
        residuals.append(residual)

    return x_vals, residuals


def GKTH_self_consistency_1S_find_root(
    p: GlobalParams,
    layers: List[Layer],
    layers_to_check: list[int] = [0],
):
    """Finds Delta for a single superconducting layer or several identical superconducting layers.

    Args:
        p (GlobalParams): Global parameter object defining the stack and calculation parameters.
        layers (List[Layer]): An array of Layer objects, defining the junction structure.
        layers_to_check (list[int], optional): An array of integers defining which layers in the stack should have Delta calculated. Defaults to [0].

    Raises:
        RuntimeError: If initial Delta_0 value is too small

    Returns:
        Delta: The gap calculated after running self-consistency.
        layers: The array of Layer objects with updated Delta values.
    """
    tol = p.abs_tolerance_self_consistency_1S
    min_Delta = tol
    max_Delta = layers[layers_to_check[0]].Delta_0

    # Root-finding process
    x0 = (min_Delta + max_Delta) / 2
    sol = root_scalar(
        lambda x: GKTH_self_consistency_1S_residual(x, p, layers, layers_to_check),
        x0=x0,
        xtol=tol,
    )
    Delta = sol.root
    if not sol.converged:
        raise RuntimeError("Root-finding did not converge")

    # Set the calculated Delta for the layers being checked
    for j in layers_to_check:
        layers[j].Delta_0 = Delta

    print(f"Solution Delta({p.T}, {p.h}) = {Delta} eV")

    return Delta, layers
