from copy import deepcopy
from typing import List

import numpy as np
from scipy.integrate import trapezoid

from Global_Parameter import GlobalParams
from Green_Function import GKTH_find_radial_ks, GKTH_hamiltonian_k
from Layer import Layer


def GKTH_Greens_current_radial(
    p: GlobalParams, layers: List[Layer], include_spin=False, **kwargs
):
    """
    Inverts the Hamiltonian and takes a sum over k1, k2 and Matsubara frequencies,
    returning the current between each layer.

    Parameters:
    -----------
    p : object
        Global Parameter object defining the stack and calculation parameters.
    layers : list
        An array of Layer objects defining the junction structure.

    Optional Parameters:
    -------------------
    include_spin : bool
        Whether to include spin components (default: False)
    minCalcs : int
        Minimum number of calculations (default: 50)
    maxCalcs : int
        Maximum number of calculations (default: 500)
    maxMatsubara : float
        Maximum Matsubara frequency (default: 1e7)
    layers_to_check : list
        Which layers to check (default: [0, 2])

    Returns:
    --------
    js : ndarray
        Current between each pair of layers
    E_kresolved_matsum : ndarray, optional
        Matrix sum over k-space (if verbose=True)
    k1s : ndarray
        k1 points used in calculation
    k2s : ndarray
        k2 points used in calculation
    new_rs : ndarray
        Radial points
    radial_angles : ndarray
        Angular points
    """

    def calculate_ksum(n, base_m):
        # Matrix for adding frequency dependence
        imaginary_identity_m = 1j * np.eye(4 * nlayers, 4 * nlayers)

        # Calculate the Matsubara frequency value
        w = (2 * n + 1) * np.pi * p.T
        ws = imaginary_identity_m * w

        # Go through each k1,k2 point, invert the Hamiltonian to find Fupdown and Fdownup
        ws = np.tile(ws, (nrs, nangles, 1, 1))
        base_m = np.transpose(base_m, (2, 3, 0, 1))
        E_matrices_kresolved = np.linalg.inv(base_m + ws) + np.linalg.inv(base_m - ws)

        # Finding the new E matrix
        E_matrices = np.zeros((2, 2, ninterfaces), dtype=complex)

        for i in range(ninterfaces):
            x_idx = np.mod([Expos[i], Expos[i] + 1], E_matrices_kresolved.shape[2])
            y_idx = np.mod([Eypos[i], Eypos[i] + 1], E_matrices_kresolved.shape[3])
            patch = E_matrices_kresolved[:, :, x_idx[:, None], y_idx]
            E_matrices[:, :, i] = (
                np.einsum("abjk,ab->jk", patch, area_factor) * normalisation_factors[i]
            )

        j_txyz = np.zeros(ninterfaces * 4)
        j_txyz[::4] = np.imag(np.trace(E_matrices, axis1=0, axis2=1))
        j_txyz[1::4] = np.imag(np.trace(px @ E_matrices, axis1=0, axis2=1))
        j_txyz[2::4] = np.imag(np.trace(py @ E_matrices, axis1=0, axis2=1))
        j_txyz[3::4] = np.imag(np.trace(pz @ E_matrices, axis1=0, axis2=1))

        return j_txyz

    # Handle optional parameters with default values
    maxCalcs = kwargs.get("maxCalcs", 500)
    maxMatsubara = kwargs.get("maxMatsubara", 1e7)
    layers_to_check = kwargs.get("layers_to_check", [0, 2])

    # Override maxMatsubara as per original code
    maxMatsubara = 1e6 + 1e4 / p.T

    # Initializing values
    nlayers = len(layers)
    if nlayers < 2:
        raise ValueError("Can't calculate current with <2 layers")

    ninterfaces = nlayers - 1 + p.cyclic_tunnelling

    # Get the k-points (assuming this function has been converted)
    k1s, k2s, new_rs, radial_angles, area_factor = GKTH_find_radial_ks(
        deepcopy(p),
        deepcopy(layers),
        width=abs(p.ts[0]) ** 0.5,
        just_use_layer=layers_to_check,
    )

    nrs, nangles = k1s.shape

    # Prefactor: 2 * e * t * T * (2pi/a)^2 / hbar
    normalisation_factors = (
        -2 * 1.60217e-19 * p.ts * p.T * (2 * np.pi / p.a) ** 2 / 6.582119569e-16
    )
    # Multiplied by E (eV-1) gives C m-2 s-1 = current density

    # Building the base matrices with no matsubara frequency dependence
    base_m = GKTH_hamiltonian_k(p, k1s, k2s, layers)

    # Pauli matrices
    px = np.array([[0, 1], [1, 0]])
    py = np.array([[0, -1j], [1j, 0]])
    pz = np.array([[1, 0], [0, -1]])

    if include_spin:
        j_to_include = np.ones(4 * ninterfaces)
    else:
        j_to_include = np.array([i % 4 == 0 for i in range(4 * ninterfaces)], dtype=int)

    # Convert 1-based indexing to 0-based for array positions
    Expos = 4 * np.arange(1, ninterfaces + 1)
    Eypos = Expos - 4

    if p.cyclic_tunnelling:
        Expos[ninterfaces - 1] = 4 * (ninterfaces - 1)
        Eypos[ninterfaces - 1] = 0

    # For the first manually defined frequencies
    first_freqs = np.array([0, maxMatsubara])
    L = len(first_freqs)

    # Preallocate array size for speed
    ksums = np.zeros((maxCalcs, ninterfaces * 4))
    matsubara_freqs = np.zeros(maxCalcs)
    weights = np.zeros(maxCalcs)
    diffs = np.zeros(maxCalcs - 1)
    ddiffs = np.zeros(maxCalcs - 1)
    matsdiffs = np.zeros(maxCalcs - 1)

    # Calculate ksums for first frequencies
    matsubara_freqs[:L] = first_freqs
    for itr in range(L):
        ksums[itr, :] = calculate_ksum(matsubara_freqs[itr], base_m)
        weights[itr] = np.linalg.norm(ksums[itr, :] * j_to_include)

    # Iterate, dynamically sample new points based on weighting
    prev_tol_check = 0
    stopped_changing = 0

    for itr in range(L, maxCalcs):
        # Calculate all the differentials
        diffs[: itr - 1] = np.diff(weights[:itr])

        if len(diffs[: itr - 1]) >= 2:
            ddiffs[: itr - 1] = np.gradient(diffs[: itr - 1])
        else:
            ddiffs[: itr - 1] = 0

        matsdiffs[: itr - 1] = np.diff(matsubara_freqs[:itr])

        # Don't try to sample between points already next to each other
        checks = matsdiffs[: itr - 1] > 1

        # Find the new index from the max of the weighting
        mats_idx = np.argmax(np.abs(checks * ddiffs[: itr - 1] * matsdiffs[: itr - 1]))
        new_n = (matsubara_freqs[mats_idx] + matsubara_freqs[mats_idx + 1]) // 2

        # Stick the new n (sorted) into frequency array
        matsubara_freqs = np.concatenate(
            [
                matsubara_freqs[: mats_idx + 1],
                [new_n],
                matsubara_freqs[mats_idx + 1 : itr],
            ]
        )
        new_ksum = calculate_ksum(new_n, base_m)
        ksums = np.concatenate(
            [ksums[: mats_idx + 1], [new_ksum], ksums[mats_idx + 1 : itr]]
        )

        weights = np.concatenate(
            [
                weights[: mats_idx + 1],
                [np.linalg.norm(new_ksum * j_to_include)],
                weights[mats_idx + 1 : itr],
            ]
        )

        # Every 10 iterations check for convergence
        if (itr + 1) % 10 == 0:
            tol_check = (
                trapezoid(weights[: itr + 1], matsubara_freqs[: itr + 1])
                + 0.5 * weights[0]
            )
            if (
                abs((tol_check - prev_tol_check) / prev_tol_check)
                < p.rel_tolerance_Greens
            ):
                stopped_changing += 1
                if stopped_changing == 2:
                    break
            else:
                stopped_changing = 0
            prev_tol_check = tol_check

    # Finally, integrate over the whole thing to approximate the sum from 0 to
    # infinity. You need to add half the first step as trapz cuts this off.
    js = np.zeros((ninterfaces, 4))
    for interface in range(ninterfaces):
        for component in range(4):
            idx = 4 * interface + component

            # TODO: Check this: Matlab convention is (x, y) but Python is (y, x).
            # But for some reason it is interpreted as (x, y) here?
            # (Leaving the constant inside and outside of the trapezoid function
            # produces different results.)
            js[interface, component] = -trapezoid(
                matsubara_freqs[: itr + 1], ksums[: itr + 1, idx] + 0.5 * ksums[0, idx]
            )
            # print(
            #     trapezoid(matsubara_freqs[: itr + 1], ksums[: itr + 1, idx])
            #     + 0.5 * ksums[0, idx]
            # )

    return js, None, k1s, k2s, new_rs, radial_angles
