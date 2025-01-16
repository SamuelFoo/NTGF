import numpy as np

from Global_Parameter import GlobalParams


def GKTH_Delta(p: GlobalParams, symmetry: str, Delta: np.ndarray):
    """
    Description:
    GKTH_Delta builds the gap k-dependence, depending on the gap size and
    symmetry. Uses the square grid of k points defined in p.

    Inputs:
        p: Global Parameter object defining the stack and calculation parameters.
        L: A Layer object, the layer the gap should be calculated for.
        Delta: Double, the size of the gap.

    Outputs:
        An nxn matrix (dependent on nkpoints in p) with the size of the gap at each k1, k2 pair
    """

    # Create a grid of k-points
    if symmetry == "d":
        k1, k2 = np.meshgrid(p.k1 * p.a, p.k2 * p.a)
        return Delta / 2 * (np.cos(k1) - np.cos(k2))
    elif symmetry == "s":
        return np.full((p.nkpoints, p.nkpoints), Delta)
    elif symmetry == "n":
        return np.zeros((p.nkpoints, p.nkpoints))
    else:
        raise ValueError(
            "Input a symmetry for the gap. s = s-wave, d = d-wave, n=non-superconducting. p-wave not supported yet."
        )
