from typing import List

import numpy as np
from scipy.linalg import inv

from Global_Parameter import GlobalParams
from Green_Function import GKTH_find_radial_ks, GKTH_hamiltonian_k
from Layer import Layer


def GKTH_Greens_current_radial(p: GlobalParams, layers: List[Layer], **kwargs):
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
    verbose : bool
        Whether to output detailed information (default: False)
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

    # Define the nested function for calculating ksum
    def calculate_ksum(n):
        # Calculate the Matsubara frequency value
        w = (2 * n + 1) * np.pi * p.T
        ws = imaginary_identity_m * w
        E_matrices_kresolved = np.zeros(
            (nrs, nangles, 4 * nlayers, 4 * nlayers), dtype=complex
        )

        # Go through each k1,k2 point, invert the Hamiltonian to find Fupdown and Fdownup
        for i1 in range(nrs):
            for i2 in range(nangles):
                E_matrices_kresolved[i1, i2, :, :] = inv(
                    base_m[:, :, i1, i2] + ws
                ) + inv(base_m[:, :, i1, i2] - ws)

        # Finding the new E matrix
        E_matrices = np.zeros((2, 2, ninterfaces), dtype=complex)
        j_txyz = np.zeros(ninterfaces * 4)

        for i in range(ninterfaces):
            for j in range(2):
                for k in range(2):
                    E_matrices[j, k, i] = (
                        np.sum(
                            E_matrices_kresolved[
                                :, :, Expos[i] - 1 + j, Eypos[i] - 1 + k
                            ]
                            * area_factor
                        )
                        * normalisation_factors[i]
                    )

            j_txyz[4 * i] = np.imag(np.trace(E_matrices[:, :, i]))
            j_txyz[4 * i + 1] = np.imag(np.trace(px @ E_matrices[:, :, i]))
            j_txyz[4 * i + 2] = np.imag(np.trace(py @ E_matrices[:, :, i]))
            j_txyz[4 * i + 3] = np.imag(np.trace(pz @ E_matrices[:, :, i]))

        if verbose:
            nonlocal itr
            matsubara_freqs_unsrt[itr] = n
            for i1 in range(nrs):
                for i2 in range(nangles):
                    for i in range(ninterfaces):
                        for j in range(2):
                            for k in range(2):
                                E_matrices[j, k, i] = E_matrices_kresolved[
                                    i1, i2, Expos[i] - 1 + j, Eypos[i] - 1 + k
                                ]

                        E_kresolved[0, itr, i, i1, i2] = np.imag(
                            np.trace(E_matrices[:, :, i])
                        )
                        E_kresolved[1, itr, i, i1, i2] = np.imag(
                            np.trace(px @ E_matrices[:, :, i])
                        )
                        E_kresolved[2, itr, i, i1, i2] = np.imag(
                            np.trace(py @ E_matrices[:, :, i])
                        )
                        E_kresolved[3, itr, i, i1, i2] = np.imag(
                            np.trace(pz @ E_matrices[:, :, i])
                        )

        return j_txyz

    # Handle optional parameters with default values
    include_spin = kwargs.get("include_spin", False)
    minCalcs = kwargs.get("minCalcs", 50)
    maxCalcs = kwargs.get("maxCalcs", 500)
    maxMatsubara = kwargs.get("maxMatsubara", 1e7)
    verbose = kwargs.get("verbose", False)
    layers_to_check = kwargs.get(
        "layers_to_check", [0, 2]
    )  # Convert to 0-based indexing from MATLAB's [1, 3]

    # Override maxMatsubara as per original code
    maxMatsubara = 1e6 + 1e4 / p.T

    # Initializing values
    nlayers = len(layers)
    if nlayers < 2:
        raise ValueError("Can't calculate current with <2 layers")

    ninterfaces = nlayers - 1 + p.cyclic_tunnelling

    # Get the k-points (assuming this function has been converted)
    k1s, k2s, new_rs, radial_angles, area_factor = GKTH_find_radial_ks(
        p, layers, width=abs(p.ts[0]) ** 0.5, just_use_layer=layers_to_check
    )

    nrs, nangles = k1s.shape

    # Prefactor: 2 * e * t * T * (2pi/a)^2 / hbar
    normalisation_factors = (
        -2 * 1.60217e-19 * p.ts * p.T * (2 * np.pi / p.a) ** 2 / 6.582119569e-16
    )
    # Multiplied by E (eV-1) gives C m-2 s-1 = current density

    # Building the base matrices with no matsubara frequency dependence
    base_m = GKTH_hamiltonian_k(p, k1s, k2s, layers)

    # Matrix for adding frequency dependence
    imaginary_identity_m = 1j * np.eye(4 * nlayers, 4 * nlayers)

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
    values = np.zeros((maxCalcs, 2))
    integrals = np.zeros(maxCalcs - L)

    # For verbose only
    matsubara_freqs_unsrt = np.zeros(maxCalcs)
    E_kresolved = np.zeros((4, maxCalcs, ninterfaces, nrs, nangles))
    E_kresolved_matsum = np.zeros((4, ninterfaces, nrs, nangles))

    # Calculate ksums for first frequencies
    matsubara_freqs[:L] = first_freqs
    for itr in range(L):
        ksums[itr, :] = calculate_ksum(matsubara_freqs[itr])
        weights[itr] = np.linalg.norm(ksums[itr, :] * j_to_include)
        if verbose:
            values[itr, :] = [matsubara_freqs[itr], weights[itr]]

    # Iterate, dynamically sample new points based on weighting
    prev_tol_check = 0
    stopped_changing = 0

    for itr in range(L, maxCalcs):
        # Calculate all the differentials
        diffs[: itr - 1] = np.diff(weights[:itr])
        ddiffs[: itr - 1] = np.gradient(diffs[: itr - 1])
        matsdiffs[: itr - 1] = np.diff(matsubara_freqs[:itr])

        # Don't try to sample between points already next to each other
        checks = matsdiffs[: itr - 1] > 1

        # Find the new index from the max of the weighting
        weighted_diffs = np.abs(checks * ddiffs[: itr - 1] * matsdiffs[: itr - 1])
        if np.any(weighted_diffs):  # Check if any non-zero values
            mats_idx = np.argmax(weighted_diffs)
        else:
            # Handle case where all values are zero
            break

        new_n = int((matsubara_freqs[mats_idx] + matsubara_freqs[mats_idx + 1]) // 2)

        # Stick the new n (sorted) into frequency array
        matsubara_freqs = np.concatenate(
            [
                matsubara_freqs[: mats_idx + 1],
                [new_n],
                matsubara_freqs[mats_idx + 1 : itr],
            ]
        )

        # Calculate the sum over k for the new frequency
        new_ksum = calculate_ksum(new_n)

        # Stick this into ksum array
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

        # If verbose, track the integral every step
        if verbose:
            integrals[itr - L] = (
                np.trapz(matsubara_freqs[: itr + 1], weights[: itr + 1])
                + 0.5 * weights[0]
            )
            values[itr, :] = [new_n, np.linalg.norm(new_ksum * j_to_include)]

        # Every 10 iterations check for convergence
        if itr % 10 == 0 and itr > minCalcs:
            tol_check = (
                np.trapz(matsubara_freqs[: itr + 1], weights[: itr + 1])
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
            js[interface, component] = (
                np.trapz(matsubara_freqs[: itr + 1], ksums[: itr + 1, idx])
                + 0.5 * ksums[0, idx]
            )

    if verbose:
        sorted_idx = np.argsort(matsubara_freqs_unsrt[: itr + 1])
        E_kresolved = E_kresolved[:, sorted_idx, :, :, :]

        for k in range(4):
            for j in range(ninterfaces):
                for i1 in range(nrs):
                    for i2 in range(nangles):
                        E_kresolved_matsum[k, j, i1, i2] = (
                            np.trapz(
                                matsubara_freqs[: itr + 1],
                                E_kresolved[k, : itr + 1, j, i1, i2],
                            )
                            + 0.5 * E_kresolved[k, 0, j, i1, i2]
                        )

    return js, E_kresolved_matsum, k1s, k2s, new_rs, radial_angles
