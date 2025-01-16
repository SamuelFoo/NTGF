import numpy as np

from Delta import GKTH_Delta


class Layer:
    # Description
    # A Layer object contains all of its properties which are used to
    # define its electronic spectrum, gap, and its contribution to the
    # overall Hamiltonian.

    def __init__(self, n=None):
        # Superconductivity
        self.Delta_0 = 0.01  # Delta_0
        self.symmetry = (
            "s"  # Symmetry of the superconductor: s=swave d=dwave or n=normal
        )
        self._lambda = 0.1  # Superconducting coupling strength
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
        self.tNN = -0.1523  # Nearest neighbour hopping parameter in eV
        self.tNNN = 0  # Next-nearest neighbour hopping parameter in eV
        self.mu = 0.1025  # Chemical potential in eV
        self.dispersion_type = "tb"  # Type of dispersion

        # Handle array creation if n is provided
        if n is not None:
            return [Layer() for _ in range(n)]

    # Superconducting gap
    def Ds(self, p):
        # Assume GKTH_Delta is a function defined elsewhere that computes the superconducting gap
        return GKTH_Delta(p, self, self.Delta_0)

    # Electronic spectrum
    def xis(self, p):
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
