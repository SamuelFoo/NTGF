from typing import List

import numpy as np
from Global_Parameter import GlobalParams
from Layer import Layer


def rotate_y(m, theta):
    """Rotation around y-axis"""
    Ry = np.array(
        [
            [np.cos(theta / 2), np.sin(theta / 2)],
            [-np.sin(theta / 2), np.cos(theta / 2)],
        ]
    )
    return Ry @ m @ Ry.T.conj()


def rotate_z(m, theta_ip):
    """Rotation around z-axis"""
    Rz = np.array([[np.exp(1j * theta_ip / 2), 0], [0, np.exp(-1j * theta_ip / 2)]])
    return Rz @ m @ Rz.T.conj()


def GKTH_spectrum_k(p: GlobalParams, L: Layer, k1s: np.ndarray, k2s: np.ndarray):
    """
    Computes the electronic dispersion for any given k values defined by k1s and k2s.

    Parameters:
        p   : Dictionary containing global parameters (e.g., lattice constant `a`)
        L   : Dictionary containing layer properties (`dispersion_type`, `mu`, `tNN`, `tNNN`)
        k1s : 2D NumPy array of kx points
        k2s : 2D NumPy array of ky points

    Returns:
        xis : 2D NumPy array containing the electronic dispersion values at each k-point
    """

    if L.dispersion_type == "tb":  # Tight-binding dispersion
        xis = (
            L.tNN * (np.cos(k1s * p.a) + np.cos(k2s * p.a))
            + L.tNNN * np.cos(k1s * p.a) * np.cos(k2s * p.a)
            + L.mu
        )

    elif L.dispersion_type == "para":  # Parabolic dispersion
        hbar = 6.582119569e-16  # Reduced Planck’s constant (eV·s)
        me = 9.10938356e-31  # Electron mass (kg)
        xis = (hbar**2 * ((k1s * p.a) ** 2 + (k2s * p.a) ** 2) / (2 * me)) + L.mu

    else:
        raise ValueError(
            "Spectrum type must be 'tb' for tight binding or 'para' for parabolic."
        )

    return xis


def GKTH_Delta_k(p: GlobalParams, L: Layer, k1s: np.ndarray, k2s: np.ndarray):
    """
    Computes the superconducting gap (Δ) as a function of k-space,
    based on symmetry and gap size.

    Parameters:
        p   : GlobalParams object (must have attribute 'a' for lattice constant)
        L   : Layer object (must have attributes 'symmetry' and 'Delta_0')
        k1s : 2D NumPy array of kx points
        k2s : 2D NumPy array of ky points

    Returns:
        Ds  : 2D NumPy array of gap values, same shape as k1s/k2s
    """

    if L.symmetry == "d":  # d-wave symmetry
        Ds = (L.Delta_0 / 2) * (np.cos(k1s * p.a) - np.cos(k2s * p.a))

    elif L.symmetry == "s":  # s-wave symmetry
        Ds = np.full_like(k1s, L.Delta_0)

    elif L.symmetry == "n":  # Non-superconducting layer
        Ds = np.zeros_like(k1s)

    else:
        raise ValueError(
            "Invalid symmetry type. Use 's' for s-wave, 'd' for d-wave, 'n' for non-superconducting. p-wave not yet supported."
        )

    return Ds


def GKTH_hamiltonian(p: GlobalParams, layers: List[Layer]):
    """
    %Description
    Makes a (4*nlayers) * (4*nlayers) * nkpoints * nkpoints matrix which defines
    the hamiltonian at every k-point defined in the square grid by the Global
    Params p.

    %Inputs
    p           :   Global Parameter object defining the stack and
                    calculation paramters
    layers      :   An array of Layer objects, defining the junction
                    structure.

    %Outputs
    m           :   The (4*nlayers) * (4*nlayers) * nkpoints * nkpoints Hamiltonian matrix

    """
    # Create base matrix
    nlayers = len(layers)
    m = np.zeros((4 * nlayers, 4 * nlayers, p.nkpoints, p.nkpoints), dtype=complex)

    for i in range(nlayers):
        L = layers[i]
        idx = i * 4

        # Electronic Spectrum
        xis = np.zeros((2, 2, p.nkpoints, p.nkpoints), dtype=complex)
        xis[0, 0, :, :] = L.xis(p)
        xis[1, 1, :, :] = L.xis(p)

        # Expand h_layer and h_global to match xis shape
        temp_matrix = np.array([[-L.h, 0], [0, L.h + L.dE]], dtype=complex)
        h_layer = rotate_z(rotate_y(temp_matrix, L.theta), L.theta_ip)
        h_layer = np.repeat(h_layer[:, :, np.newaxis, np.newaxis], p.nkpoints, axis=2)
        h_layer = np.repeat(h_layer, p.nkpoints, axis=3)

        temp_matrix = np.array([[-p.h, 0], [0, p.h]], dtype=complex)
        h_global = rotate_z(rotate_y(temp_matrix, p.theta), p.theta_ip)
        h_global = np.repeat(h_global[:, :, np.newaxis, np.newaxis], p.nkpoints, axis=2)
        h_global = np.repeat(h_global, p.nkpoints, axis=3)

        # SOC
        if L.alpha != 0:
            SOC = np.zeros((2, 2, p.nkpoints, p.nkpoints), dtype=complex)

            if p.interface_normal == 1:
                SOC[0, 0, :, :] = -p.k1
                SOC[0, 1, :, :] = -1j * p.k2
                SOC[1, 0, :, :] = 1j * p.k2
                SOC[1, 1, :, :] = p.k1
            elif p.interface_normal == 2:
                SOC[0, 0, :, :] = p.k2
                SOC[0, 1, :, :] = -p.k1
                SOC[1, 0, :, :] = -p.k1
                SOC[1, 1, :, :] = -p.k2
            elif p.interface_normal == 3:
                SOC[0, 1, :, :] = 1j * p.k1 + p.k2
                SOC[1, 0, :, :] = -1j * p.k1 + p.k2

            SOC *= -L.alpha * p.a / np.pi
        else:
            SOC = 0

        # Single-particle matrix
        m[idx : idx + 2, idx : idx + 2, :, :] = xis + h_layer + h_global + SOC
        m[idx + 2 : idx + 4, idx + 2 : idx + 4, :, :] = -np.conj(
            xis + h_layer + h_global + SOC
        )

        # Superconductivity matrix
        GTKH_Ds = L.Ds(p)
        m[idx, idx + 3, :, :] = np.exp(1j * L.phi) * GTKH_Ds
        m[idx + 1, idx + 2, :, :] = -np.exp(1j * L.phi) * GTKH_Ds
        m[idx + 2, idx + 1, :, :] = -np.exp(-1j * L.phi) * GTKH_Ds
        m[idx + 3, idx, :, :] = np.exp(-1j * L.phi) * GTKH_Ds

    # Tunneling between layers
    signs = [1, 1, -1, -1]
    for i in range(nlayers - 1):
        idx = i * 4
        for j in range(4):
            m[idx + j, idx + 4 + j, :, :] = p.ts[i] * signs[j]
            m[idx + 4 + j, idx + j, :, :] = p.ts[i] * signs[j]

    if p.cyclic_tunnelling:
        shift = 4 * (nlayers - 1)
        for j in range(4):
            m[j, j + shift, :, :] = p.ts[i] * signs[j]
            m[j + shift, j, :, :] = p.ts[i] * signs[j]
    return m


def GKTH_hamiltonian_k(
    p: GlobalParams, k1s: np.ndarray, k2s: np.ndarray, layers: List[Layer]
):
    """
    Constructs the (4*nlayers)x(4*nlayers) Hamiltonian matrix for every pair
    of k-points (kx, ky) defined by k1s and k2s.

    Parameters:
        p       : Global parameter dictionary
        k1s     : 2D NumPy array of kx points
        k2s     : 2D NumPy array of ky points
        layers  : List of Layer objects defining the junction structure

    Returns:
        m       : Hamiltonian matrix for every k-point
    """

    nlayers = len(layers)
    shape = k1s.shape
    m = np.zeros((4 * nlayers, 4 * nlayers) + shape, dtype=complex)

    # Fill diagonal blocks for each layer
    for i, layer in enumerate(layers):
        idx = i * 4

        # Electronic spectrum
        xis = np.zeros((2, 2) + shape, dtype=complex)
        xis[0, 0, :, :] = -GKTH_spectrum_k(p, layer, k1s, k2s)
        xis[1, 1, :, :] = -GKTH_spectrum_k(p, layer, k1s, k2s)

        # Magnetism
        h_layer = rotate_z(
            rotate_y(np.array([[layer.h, 0], [0, -layer.h - layer.dE]]), -layer.theta),
            -layer.theta_ip,
        )
        h_layer = np.repeat(h_layer[:, :, np.newaxis, np.newaxis], shape[0], axis=2)
        h_layer = np.repeat(h_layer, shape[1], axis=3)

        h_global = rotate_z(
            rotate_y(np.array([[p.h, 0], [0, -p.h]]), -p.theta), -p.theta_ip
        )
        h_global = np.repeat(h_global[:, :, np.newaxis, np.newaxis], shape[0], axis=2)
        h_global = np.repeat(h_global, shape[1], axis=3)

        # Spin-Orbit Coupling (SOC)
        if layer.alpha != 0:
            SOC = np.zeros((2, 2) + shape, dtype=complex)
            if p.interface_normal == 1:
                SOC[0, 0, :, :] = -k1s
                SOC[0, 1, :, :] = -1j * k2s
                SOC[1, 0, :, :] = 1j * k2s
                SOC[1, 1, :, :] = k1s
            elif p.interface_normal == 2:
                SOC[0, 0, :, :] = k2s
                SOC[0, 1, :, :] = -k1s
                SOC[1, 0, :, :] = -k1s
                SOC[1, 1, :, :] = -k2s
            elif p.interface_normal == 3:
                SOC[0, 1, :, :] = 1j * k1s + k2s
                SOC[1, 0, :, :] = -1j * k1s + k2s
            SOC *= -layer.alpha * p.a / np.pi
        else:
            SOC = 0

        m[idx : idx + 2, idx : idx + 2, :, :] = xis + h_layer + h_global + SOC
        m[idx + 2 : idx + 4, idx + 2 : idx + 4, :, :] = -np.conj(
            xis + h_layer + h_global + SOC
        )

        # Superconductivity
        Delta_k = GKTH_Delta_k(p, layer, k1s, k2s)
        phi = layer.phi
        m[idx, idx + 3, :, :] = np.exp(1j * phi) * Delta_k
        m[idx + 1, idx + 2, :, :] = -np.exp(1j * phi) * Delta_k
        m[idx + 2, idx + 1, :, :] = -np.exp(-1j * phi) * Delta_k
        m[idx + 3, idx, :, :] = np.exp(-1j * phi) * Delta_k

    # Tunneling between layers
    signs = [-1, -1, 1, 1]
    for i in range(nlayers - 1):
        idx = i * 4
        for j in range(4):
            m[idx + j, idx + j + 4, :, :] = p.ts[i] * signs[j]
            m[idx + j + 4, idx + j, :, :] = p.ts[i] * signs[j]

    # Cyclic tunneling
    if p.cyclic_tunnelling:
        shift = 4 * (nlayers - 1)
        for j in range(4):
            m[j, j + shift, :, :] = p.ts[-1] * signs[j]
            m[j + shift, j, :, :] = p.ts[-1] * signs[j]

    return m
