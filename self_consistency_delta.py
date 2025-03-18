from typing import List

import numpy as np
from scipy.optimize import root, root_scalar

from Global_Parameter import GlobalParams
from Green_Function import GKTH_Greens, GKTH_Greens_radial
from Layer import Layer


# Residual function for self-consistency
def GKTH_self_consistency_1S_residual(
    Delta_0_fit: float, p: GlobalParams, layers: List[Layer], layers_to_check: List[int]
) -> np.float64:
    # Update Delta_0 for layers being checked
    for i in layers_to_check:
        layers[i].Delta_0 = Delta_0_fit

    # Calculate the anomalous Green function sums
    Fs_sums = GKTH_Greens(p, layers)
    idx = layers_to_check[0]
    residual = layers[idx]._lambda * p.T * np.abs(Fs_sums[idx]) - Delta_0_fit
    # print(f"delta_0 residual: {Delta_0_fit:.3e} {residual:.4e}")
    return residual


def GKTH_self_consistency_1S_iterate(
    p: GlobalParams,
    layers: List[Layer],
    layers_to_check: list[int] = [0],
    max_Delta: float = 0.002,
):
    x_vals = np.linspace(0, max_Delta, 50)
    residuals = []
    for x in x_vals:
        residual = GKTH_self_consistency_1S_residual(
            Delta_0_fit=x, p=p, layers=layers, layers_to_check=layers_to_check
        )
        residuals.append(residual.item())

    return x_vals, residuals


def GKTH_self_consistency_1S_find_root(
    p: GlobalParams,
    layers: List[Layer],
    layers_to_check: list[int] = [0],
    max_Delta: float = 0.002,
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


def GKTH_self_consistency_2S_taketurns(
    p: GlobalParams, layers: List[Layer], layers_to_check: List[int] = [0, 1]
):
    """
    Description:
    GKTH_self_consistency_2S uses self-consistency to find the Delta values
    for two different superconductors in a stack. This approach is rougher
    than for a single superconductor, making it more prone to error, and
    it will not detect first-order transitions.

    The method applies a numerical Newton-Raphson approach to approximate
    the self-consistent solution for one superconductor while keeping the
    other superconductor's gap constant. The process is then repeated
    for the second superconductor, iterating until convergence within
    a given tolerance. This approach appears more stable than gradient descent.

    Inputs:
        p               : Global parameter object defining the stack and
                          calculation parameters.
        layers          : A list of Layer objects defining the junction structure.
        layers_to_check : A list of indices specifying which layers should
                          have their Delta values calculated.

    Outputs:
        Delta  : The computed gap values after self-consistency iteration.
        layers : The updated list of Layer objects with updated Delta values.
        history: A matrix storing checked Delta values along with the
                 changes in Delta from the self-consistency calculations.
                 Useful for analyzing algorithm progress.
    """

    tol = p.abs_tolerance_self_consistency_1S
    history = []

    def GKTH_self_consistency_2S_residual2D(x):
        """
        Computes the residual between the calculated Delta values and the
        expected ones based on self-consistency.
        """
        for i in layers_to_check:
            layers[i].Delta_0 = x[i]

        Fs_sums, _, _, _ = GKTH_Greens_radial(p, layers, layers_to_check=[0, 1])
        Deltas_iterate = (
            np.array([layer._lambda for layer in layers]) * p.T * np.abs(Fs_sums)
        )
        residual2D = np.real(Deltas_iterate - x)

        history.append((x.copy(), residual2D.copy()))
        return residual2D

    # Initialize x with the current Delta_0 values of the specified layers
    x = np.array([layers[i].Delta_0 for i in layers_to_check], dtype=np.float64)

    sol = root(
        GKTH_self_consistency_2S_residual2D,
        x0=x,
        tol=tol,
        method="krylov",
    )
    Delta = sol.x
    if not sol.success:
        raise RuntimeError("Root-finding did not converge", sol.message)
    print("Number of function evaluations:", sol.nfev)

    # Update the Delta_0 values in the layers being checked
    for j, layer_index in enumerate(layers_to_check):
        layers[layer_index].Delta_0 = np.real(Delta[j])

    return Delta, layers, history
