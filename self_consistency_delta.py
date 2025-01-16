from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar

from Global_Parameter import GlobalParams
from Green_Function import GKTH_Greens
from Layer import Layer


def GKTH_self_consistency_1S(
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

    # Residual function for self-consistency
    def GKTH_self_consistency_1S_residual(Delta_0_fit):
        # Update Delta_0 for layers being checked
        for i in layers_to_check:
            layers[i].Delta_0 = Delta_0_fit
        # Calculate the anomalous Green function sums
        Fs_sums = GKTH_Greens(p, layers)
        idx = layers_to_check[0]
        residual = layers[idx]._lambda * p.T * np.abs(Fs_sums[idx]) - Delta_0_fit
        print(f"delta_0 residual: {Delta_0_fit:.3e} {residual:.4e}")
        return residual

    # Root-finding process
    x0 = (min_Delta + max_Delta) / 2
    sol = root_scalar(GKTH_self_consistency_1S_residual, x0=x0, xtol=tol)
    Delta = sol.root
    if not sol.converged:
        raise RuntimeError("Root-finding did not converge")

    # Set the calculated Delta for the layers being checked
    for j in layers_to_check:
        layers[j].Delta_0 = Delta

    print(f"Solution Delta({p.T}, {p.h}) = {Delta} eV")

    x_vals = np.linspace(0, 0.002, 50)
    residuals = []
    for x in x_vals:
        res = GKTH_self_consistency_1S_residual(x)
        residuals.append(res)

    plt.figure(figsize=(12, 6))
    plt.scatter(x_vals, residuals)
    plt.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Delta_0 (eV)")
    plt.ylabel("Residual (eV)")
    plt.title("Residual vs Delta_0")
    plt.show()

    return Delta, layers
