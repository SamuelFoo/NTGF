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


class Layer:
    # Description
    # A Layer object contains all of its properties which are used to
    # define its electronic spectrum, gap, and its contribution to the
    # overall Hamiltonian.

    def __init__(self, _lambda, symmetry="s", n=None):
        # Superconductivity
        self.Delta_0 = 0.0  # Delta_0
        # 1D values
        # self.Delta_0 = 0.01  # Delta_0

        # Symmetry of the superconductor: s=swave d=dwave or n=normal
        self.symmetry = symmetry
        self._lambda = _lambda  # Superconducting coupling strength
        self.phi = 0  # Superconducting phase

        # Ferromagnetism
        self.h = 0  # Symmetric exchange field in eV acting on both up and down
        self.dE = 0  # Asymmetric exchange shift of only one band in eV
        self.theta = 0  # Angle of exchange field about y, away from quantization axis z
        self.theta_ip = 0  # Angle of exchange field about z, the quantization axis

        # SOC
        self.alpha = 0

        # Electronic Spectrum
        self.N0 = 1  # Density of states at Fermi surface, just normalised to 1 for now
        self.tNN = -0.7823  # Nearest neighbour hopping parameter in eV
        self.tNNN = -0.0740  # Next-nearest neighbour hopping parameter in eV
        self.mu = 0.06525  # Chemical potential in eV
        # 1D values
        # self.tNN = -0.1523  # Nearest neighbour hopping parameter in eV
        # self.tNNN = 0  # Next-nearest neighbour hopping parameter in eV
        # self.mu = 0.1025  # Chemical potential in eV

        self.dispersion_type = "tb"  # Type of dispersion

        # Handle array creation if n is provided
        if n is not None:
            return [Layer() for _ in range(n)]

    # Superconducting gap
    def Ds(self, p: GlobalParams):
        # Assume GKTH_Delta is a function defined elsewhere that computes the superconducting gap
        return GKTH_Delta(p, self.symmetry, self.Delta_0)

    # Electronic spectrum
    def xis(self, p: GlobalParams):
        if self.dispersion_type == "tb":
            return (
                self.tNN * (np.cos(p.k1 * p.a) + np.cos(p.k2 * p.a))
                + self.tNNN * np.cos(p.k1 * p.a) * np.cos(p.k2 * p.a)
                + self.mu
            )
        elif self.dispersion_type == "para":
            hbar = 6.582119569e-16  # Planck's constant in eV.s
            me = 9.10938356e-31  # Electron mass in kg
            return (
                hbar**2 * ((p.k1 * p.a) ** 2 + (p.k2 * p.a) ** 2) / (2 * me) + self.mu
            )
        else:
            raise ValueError(
                "Spectrum type must be 'tb' for tight binding or 'para' for parabolic."
            )
