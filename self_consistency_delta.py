from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar

from Global_Parameter import GlobalParams
from Green_Function import GKTH_Greens
from Layer import Layer


def GKTH_self_consistency_1S(
    p: GlobalParams, layers: List[Layer], layers_to_check=[0], catch_first_order=True
):
    """
    GKTH_self_consistency_1S finds Delta for a single superconducting layer or several identical superconducting layers.

    Inputs:
        p : Global parameter object defining the stack and calculation parameters.
        layers : An array of Layer objects, defining the junction structure.
        layers_to_check : An array of integers defining which layers in the stack should have Delta calculated.
        catch_first_order : Boolean, whether or not to search for the first-order transition if the initial zero finding fails.

    Outputs:
        Delta : The gap calculated after running self-consistency.
        layers : The array of Layer objects with updated Delta values.
        residual_history : Array containing the checked Delta values along with the change in Delta from the self-consistency calculation.
    """
    tol = p.abs_tolerance_self_consistency_1S
    min_Delta = tol
    max_Delta = layers[layers_to_check[0]].Delta_0
    residual_history = []

    # Residual function for self-consistency
    def GKTH_self_consistency_1S_residual(Delta_0_fit):
        # Update Delta_0 for layers being checked
        for i in layers_to_check:
            layers[i].Delta_0 = Delta_0_fit
        # Calculate the anomalous Green function sums
        Fs_sums = GKTH_Greens(p, layers)
        idx = layers_to_check[0]
        residual = layers[idx]._lambda * p.T * np.abs(Fs_sums[idx]) - Delta_0_fit
        residual_history.append([Delta_0_fit, residual])
        print(f"delta_0 residual: {Delta_0_fit:.3e} {residual:.4e}")
        return residual

    # Newton-Raphson method for root-finding
    def newton_raphson(x, fx, tol, maxItr=20):
        prev_x = x
        prev_fx = fx
        x += fx
        dfdx = 0
        itr = 0
        min_neg_x = max_Delta
        while itr < maxItr:
            fx = GKTH_self_consistency_1S_residual(x)
            if fx > 0:
                try:
                    sol = root_scalar(
                        GKTH_self_consistency_1S_residual,
                        bracket=[x, min_neg_x],
                        xtol=tol,
                    )
                    x = sol.root
                except ValueError:
                    x = 0
                break
            elif fx < 0 and x < min_neg_x:
                min_neg_x = x
            dfdx = (fx - prev_fx) / (x - prev_x)
            if np.abs(fx) < tol:
                if dfdx < 0:
                    break
                else:
                    try:
                        sol = root_scalar(
                            GKTH_self_consistency_1S_residual,
                            x0=x + 2 * fx / dfdx,
                            xtol=tol,
                        )
                        x = sol.root
                    except ValueError:
                        x = 0
                        break
            prev_x = x
            prev_fx = fx
            x = prev_x - prev_fx / dfdx

            if dfdx > 0 or x < min_Delta:
                x = 0
                break
            itr += 1
            if itr == maxItr:
                x = np.nan
        return x

    # Root-finding process
    x1 = min_Delta
    x2 = max_Delta
    f1 = GKTH_self_consistency_1S_residual(x1)
    f2 = GKTH_self_consistency_1S_residual(x2)

    if f1 > 0 and f2 < 0:  # If +- then we can use interval root-finding
        sol = root_scalar(GKTH_self_consistency_1S_residual, bracket=[x1, x2], xtol=tol)
        Delta = sol.root
    elif f2 > 0:  # If ++ or -+ then need to increase max_Delta
        if f1 < 0 and not catch_first_order:
            Delta = 0
        else:
            while f2 > 0:
                x1 = x2
                f1 = f2
                x2 *= 2
                f2 = GKTH_self_consistency_1S_residual(x2)
                if f2 > 0 and x2 > 1e5:
                    raise RuntimeError(
                        "Can't find a negative change in Delta. Try increasing initial Delta_0 value."
                    )
            sol = root_scalar(
                GKTH_self_consistency_1S_residual, bracket=[x1, x2], xtol=tol
            )
            Delta = sol.root
    elif f1 < 0 and f2 < 0:  # If -- then use Newton-Raphson method
        if catch_first_order:
            print("into Newton")
            Delta = newton_raphson(max_Delta, f2, tol)
        else:
            Delta = 0
    else:
        Delta = 0
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

    return Delta, layers, residual_history
