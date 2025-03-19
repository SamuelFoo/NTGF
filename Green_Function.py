from copy import deepcopy
from typing import List, Tuple

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

from constants import kB
from Global_Parameter import GlobalParams
from Hamiltonian import GKTH_Delta_k, GKTH_hamiltonian, GKTH_hamiltonian_k
from k_space_flipping import GKTH_flipflip
from ksubsample import GKTH_ksubsample
from Layer import GKTH_Delta, Layer


def calculate_ksum(
    n,
    nlayers,
    npointscalc,
    base_m,
    imaginary_identity_m,
    compute_idxs,
    D_factors,
    overall_multiplier_flat,
    random_sampling_max,
    p,
    verbose,
    normalisation_factor,
):
    Fs_ksum = np.zeros(nlayers)
    w = (2 * n + 1) * np.pi * p.T
    ws = imaginary_identity_m * w
    Fupdowns = np.zeros((nlayers, npointscalc))
    Fdownups = np.zeros((nlayers, npointscalc))

    idx_samps = compute_idxs[:, 0] + np.round(
        random_sampling_max * np.random.rand(npointscalc)
    ).astype(int)
    +p.nkpoints * np.round(random_sampling_max * np.random.rand(npointscalc)).astype(
        int
    )

    for i in range(npointscalc):
        idx_samp = idx_samps[i]
        m_inv = np.linalg.inv(
            -base_m[:, :, compute_idxs[i, 0], compute_idxs[i, 1]] + ws
        )
        for j in range(nlayers):
            # Extract Fupdown and Fdownup for each layer
            # print(m_inv[4 * j, 4 * j + 3], m_inv[4 * j + 1, 4 * j + 2])
            Fupdowns[j, i] = (
                m_inv[4 * j, 4 * j + 3]
                * D_factors[compute_idxs[i, 0], compute_idxs[i, 1], j]
            )
            Fdownups[j, i] = (
                m_inv[4 * j + 1, 4 * j + 2]
                * D_factors[compute_idxs[i, 0], compute_idxs[i, 1], j]
            )

    # Calculate the k-space sum over all layers
    for layer in range(nlayers):
        Fs_ksum[layer] = normalisation_factor * np.sum(
            overall_multiplier_flat * (Fupdowns[layer, :] - Fdownups[layer, :])
        )

        # Store verbose data if required
        if verbose:
            F_kresolved = Fupdowns[layer, :] - Fdownups[layer, :]
            matsubara_freqs_unsrt = (
                n  # Track the current Matsubara frequency for later sorting
            )
    # print(Fs_ksum)
    return Fs_ksum


def calculate_ksum(n, base_m, imaginary_identity_m, D_factors, area_factor, p, verbose):
    """
    Computes the sum over k-points for a given Matsubara frequency n.

    Parameters:
        n                    : Matsubara frequency index
        base_m               : Precomputed Hamiltonian matrix
        imaginary_identity_m  : Identity matrix for imaginary terms
        D_factors            : Precomputed superconducting gap factors
        area_factor          : k-space area weights
        verbose              : Boolean flag for debugging information

    Returns:
        Fs_ksum: Sum over k-points
    """
    _, nrs, nangles = D_factors.shape
    w = (2 * n + 1) * np.pi * p.T

    # Invert Hamiltonian at each k-point
    ws = w * imaginary_identity_m
    ws = np.tile(ws, (nrs, nangles, 1, 1))
    base_m = np.transpose(base_m, (2, 3, 0, 1))
    m_inv = np.linalg.inv(base_m + ws)
    Fupdowns = m_inv[..., ::4, 3::4].diagonal(axis1=2, axis2=3)
    Fdownups = m_inv[..., 1::4, 2::4].diagonal(axis1=2, axis2=3)
    D_factors = np.transpose(D_factors, (1, 2, 0))
    area_factor = area_factor[..., np.newaxis]
    Fs_ksum = np.sum(area_factor * D_factors * (Fupdowns - Fdownups), axis=(0, 1))
    Fs_ksum = np.real_if_close(Fs_ksum)
    return Fs_ksum


def GKTH_Greens(
    p: GlobalParams,
    layers: List[Layer],
    density_grid=None,
    compute_grid=None,
    maxCalcs=100,
    maxMatsubara=1e7,
    verbose=False,
):
    """
    Computes the anomalous Green's function over k and Matsubara frequencies.

    Args:
    p: Parameter object
    layers: Array of layer objects defining the junction structure
    density_grid: 2D array for density grid from k-subsample (optional)
    compute_grid: 2D array for compute grid from k-subsample (optional)
    maxCalcs: Maximum number of Matsubara calculations (default 500)
    maxMatsubara: Maximum Matsubara frequency (default 1e7)
    verbose: Boolean flag to return k-resolved results and integrated values (default False)

    Returns:
    Fs_sums: Fs sum over k1, k2, and Matsubara frequencies for each layer
    matsubara_freqs: Matsubara frequencies calculated
    ksums: Total sum over k for each calculated frequency
    integrals: Integrated k-sum over Matsubara frequency (if verbose=True)
    values: Weighting function values (if verbose=True)
    F_kresolved_final: k-resolved anomalous Green function (if verbose=True)
    """
    nlayers = len(layers)
    D_factors = np.zeros((p.nkpoints, p.nkpoints, nlayers))

    for i in range(nlayers):
        D_factors[:, :, i] = 1.0 / GKTH_Delta(p, layers[i].symmetry, 1)
    D_factors[np.isnan(D_factors)] = 0

    normalisation_factor = p.k_step_size**2 / (2 * np.pi / p.a) ** 2

    # Base matrices with no Matsubara frequency dependence
    base_m = GKTH_hamiltonian(p, layers)
    imaginary_identity_m = 1j * np.eye(4 * nlayers)

    # Create matrices for k-space subsampling if not already provided
    if density_grid is None or density_grid.shape != (p.nkpoints, p.nkpoints):
        density_grid, compute_grid = GKTH_ksubsample(p, layers)

    # Flatten the 2D arrays for linear indexing
    compute_idxs = np.transpose(np.nonzero(compute_grid == 1))
    npointscalc = len(compute_idxs)
    random_sampling_max = density_grid[compute_idxs[:, 0], compute_idxs[:, 1]] - 1

    # Symmetry multiplier
    if p.lattice_symmetry == "4mm":
        symmetry_multiplier = 8 * np.ones((p.nkpoints, p.nkpoints))
        np.fill_diagonal(symmetry_multiplier, 4)
    elif p.lattice_symmetry == "mm":
        symmetry_multiplier = 4 * np.ones((p.nkpoints, p.nkpoints))
    elif p.lattice_symmetry == "m":
        symmetry_multiplier = 2 * np.ones((p.nkpoints, p.nkpoints))
    else:
        symmetry_multiplier = np.ones((p.nkpoints, p.nkpoints))

    overall_multiplier = symmetry_multiplier * density_grid**2 * compute_grid
    overall_multiplier_flat = overall_multiplier[compute_idxs[:, 0], compute_idxs[:, 1]]

    # Manually defined Matsubara frequencies
    first_freqs = [0, maxMatsubara]
    L = len(first_freqs)

    ksums = np.zeros((maxCalcs, nlayers))
    matsubara_freqs = np.zeros(maxCalcs)
    weights = np.zeros(maxCalcs)
    diffs = np.zeros(maxCalcs - 1)
    ddiffs = np.zeros(maxCalcs - 1)
    matsdiffs = np.zeros(maxCalcs - 1)
    values = np.zeros((maxCalcs, 1 + nlayers))
    integrals = np.zeros(maxCalcs - L)

    F_kresolved = np.zeros((maxCalcs, nlayers, npointscalc))
    F_kresolved_symm = np.zeros((nlayers, p.nkpoints, p.nkpoints))
    F_kresolved_final = np.zeros((nlayers, 2 * p.nkpoints, 2 * p.nkpoints))

    # Calculate ksums for the first frequencies
    matsubara_freqs[:L] = first_freqs
    for itr in range(L):
        ksums[itr, :] = calculate_ksum(
            matsubara_freqs[itr],
            nlayers,
            npointscalc,
            base_m,
            imaginary_identity_m,
            compute_idxs,
            D_factors,
            overall_multiplier_flat,
            random_sampling_max,
            p,
            verbose,
            normalisation_factor,
        )

        if verbose:
            values[itr, :] = np.concatenate(
                ([matsubara_freqs[itr]], np.abs(ksums[itr, :]))
            )
    weights[:L] = np.sum(np.abs(ksums[:L, :]), axis=1)

    # Iterate and dynamically sample new points based on weighting
    prev_tol_check = 0
    stopped_changing = 0
    for itr in range(L, maxCalcs):
        # Calculate all the differentials
        diffs[: itr - 1] = np.diff(weights[:itr])
        # Calculate the second-order gradient only when there are enough points
        if itr > 2:
            if len(diffs[: itr - 2]) >= 2:  # Ensure there are at least 2 elements
                ddiffs[: itr - 2] = np.gradient(diffs[: itr - 2])
            else:
                # Handle small cases manually or set a default value
                ddiffs[: itr - 2] = 0  # or some other default behavior
        elif itr == 2:
            ddiffs[0] = diffs[1] - diffs[0]  # Simple difference for 2 points

        matsdiffs[: itr - 1] = np.diff(matsubara_freqs[:itr])

        # Exclude points next to each other
        checks = matsdiffs[: itr - 1] > 1

        # Find the new Matsubara frequency index
        mats_idx = np.argmax(np.abs(checks * ddiffs[: itr - 1] * matsdiffs[: itr - 1]))
        new_n = (matsubara_freqs[mats_idx] + matsubara_freqs[mats_idx + 1]) // 2

        # Insert the new frequency
        matsubara_freqs = np.insert(matsubara_freqs, mats_idx + 1, new_n)

        new_ksum = calculate_ksum(
            new_n,
            nlayers,
            npointscalc,
            base_m,
            imaginary_identity_m,
            compute_idxs,
            D_factors,
            overall_multiplier_flat,
            random_sampling_max,
            p,
            verbose,
            normalisation_factor,
        )
        ksums = np.insert(ksums, mats_idx + 1, new_ksum, axis=0)
        weights = np.insert(weights, mats_idx + 1, np.sum(np.abs(new_ksum)))

        # Verbose tracking
        if verbose:
            integrals[itr - L] = (
                np.trapz(matsubara_freqs[:itr], weights[:itr]) + 0.5 * weights[0]
            )
            values[itr, :] = np.concatenate(([new_n], np.abs(new_ksum)))

        # Convergence check every 10 iterations
        if itr % 10 == 0:
            tol_check = (
                np.trapz(matsubara_freqs[:itr], weights[:itr]) + 0.5 * weights[0]
            )
            if (
                np.abs((tol_check - prev_tol_check) / prev_tol_check)
                < p.rel_tolerance_Greens
            ):
                stopped_changing += 1
                if stopped_changing == 3:
                    break
            else:
                stopped_changing = 0
            prev_tol_check = tol_check

    # Final integration
    Fs_sums = np.zeros(nlayers)
    for layer in range(nlayers):
        Fs_sums[layer] = (
            np.trapz(matsubara_freqs[:itr], ksums[:itr, layer]) + 0.5 * ksums[0, layer]
        )

    if verbose:
        for j in range(nlayers):
            for i in range(npointscalc):
                F_kresolved_symm[j, compute_idxs[i, 0], compute_idxs[i, 1]] = (
                    np.trapz(matsubara_freqs[:itr], F_kresolved[:itr, j, i])
                    + 0.5 * F_kresolved[0, j, i]
                )
            F_kresolved_final[j, :, :] = GKTH_flipflip(
                F_kresolved_symm[j, :, :],
                use_4mm_symmetry=p.use_4mm_symmetry,
                use_kspace_subsampling=p.use_kspace_subsampling,
                density_grid=density_grid,
            )

        return matsubara_freqs, ksums, F_kresolved_final

    else:
        return Fs_sums


def GKTH_find_radial_ks(
    p: GlobalParams,
    layers: List[Layer],
    base_space: float = 0.1,
    width: float = 0.1,
    just_use_layer: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Subsamples k-space radially based on proximity to the Fermi surface.
    Takes radial lines from kx = ky = 0 to the edge of the first Brillouin zone.

    Parameters:
        p       : Global parameter object
        layers  : List of Layer objects
        opts    : Dictionary with optional parameters:
            - base_space: Ratio controlling spacing along the radial direction (default 0.1)
            - width: Controls sampling sharpness near the Fermi surface (default 0.1)
            - just_use_layer: Restrict sampling to a specific layer (default 0, uses all)

    Returns:
        k1s         : Array of kx points
        k2s         : Array of ky points
        new_rs      : Array of radial distances
        radial_angles: List of angles of the radial lines
        area_factor : Array of areas of k-space represented by each k-point
    """
    # If only specific layers should be used
    if just_use_layer != 0:
        nlayers = len(just_use_layer)
        layers = [layers[i] for i in just_use_layer]
    else:
        nlayers = len(layers)

    # Initialize layer superconducting gap to zero
    for layer in layers:
        layer.Delta_0 = 0

    rs = np.zeros((p.ntest, p.nradials))
    eigenvalues = np.zeros((4 * nlayers, p.ntest, p.nradials))

    # Define radial angles based on lattice symmetry
    if p.lattice_symmetry == "4mm":
        radial_angles = np.linspace(0, (np.pi / 4) + 1e-5, p.nradials)
    elif p.lattice_symmetry == "mm":
        radial_angles = np.linspace(0, (np.pi / 2) + 1e-5, p.nradials)
    elif p.lattice_symmetry == "m":
        radial_angles = np.linspace(
            p.m_symmetry_line, p.m_symmetry_line + np.pi, p.nradials
        )
    else:
        radial_angles = np.linspace(0, 2 * np.pi, p.nradials)

    eff_angles = np.mod(radial_angles, np.pi / 2)
    eff_angles[eff_angles > np.pi / 4] -= np.pi / 2

    # Define radial distances for each angular line
    for i in range(p.nradials):
        rs[:, i] = np.linspace(0, 0.5 / np.cos(eff_angles[i]), p.ntest)

    # Compute k-space points
    k1s = (2 * np.pi / p.a) * rs * np.cos(radial_angles)
    k2s = (2 * np.pi / p.a) * rs * np.sin(radial_angles)

    # Compute Hamiltonian at each k-point
    base_m = GKTH_hamiltonian_k(p, k1s, k2s, layers)

    # Compute eigenvalues at each k-point
    for i in range(p.ntest):
        for j in range(p.nradials):
            eigenvalues[:, i, j] = np.linalg.eigvalsh(base_m[:, :, i, j])

    # Compute averaged spectrum over all eigenvalues
    avg_spectrum = np.prod(np.abs(eigenvalues), axis=0) ** (1 / (4 * nlayers))

    # Compute sampling weights based on Fermi surface proximity
    weights = (
        np.exp(-np.abs(avg_spectrum) / width)
        * rs
        * np.gradient(rs.conj().T, axis=1).conj().T
    )
    constant = np.sum(weights, axis=0) / p.ntest * base_space
    weights = constant + weights

    # Compute cumulative distribution for interpolation
    cumdist = np.cumsum(weights, axis=0)

    # Initialize new radial grid
    new_rs = np.zeros((p.nfinal, p.nradials))

    # Interpolate new radial positions based on cumulative distribution
    for j in range(p.nradials):
        interp_func = interp1d(
            cumdist[:, j], rs[:, j], kind="linear", fill_value="extrapolate"
        )
        new_rs[:, j] = interp_func(
            np.linspace(np.min(cumdist[:, j]), np.max(cumdist[:, j]), p.nfinal)
        )

    # Compute area factors
    area_factor = (
        (2 * np.pi / (p.nradials - 1))
        * new_rs
        * np.gradient(new_rs.conj().T, axis=1).conj().T
    )

    # Adjust boundary conditions for area_factor
    area_factor[:, 0] /= 2
    area_factor[:, -1] /= 2
    area_factor[0, :] /= 2
    area_factor[-1, :] /= 2

    # Compute final k-space coordinates
    k1s = (2 * np.pi / p.a) * new_rs * np.cos(radial_angles)
    k2s = (2 * np.pi / p.a) * new_rs * np.sin(radial_angles)

    return k1s, k2s, new_rs, radial_angles, area_factor


def GKTH_Greens_radial(
    p: GlobalParams,
    layers: List[Layer],
    maxCalcs=500,
    layers_to_check=[0],
    verbose=False,
    opts=None,
):
    """
    Computes the anomalous Green function by inverting the Hamiltonian and summing over
    k-space and Matsubara frequencies.

    Parameters:
        p       : Global parameter object
        layers  : List of Layer objects defining the junction structure
        maxCalcs: Maximum number of Matsubara calculations (default 500)
        verbose: Whether to return k-resolved Green function
        layers_to_check: Define which layer(s) to use for k-space sampling
        opts    : Dictionary with optional parameters:
            - maxMatsubara: Maximum Matsubara frequency (default depends on temperature)

    Returns:
        Fs_sums: Array containing the Fs sum over k1, k2, and Matsubara frequencies for each layer.
        F_kresolved_matsum (if verbose=True): k-resolved anomalous Green function summed over Matsubara frequencies.
        k1s, k2s: kx and ky points.
    """

    # Set default options
    if opts is None:
        opts = {
            "maxMatsubara": int(1e6 + 1e4 / p.T),
        }

    maxMatsubara = opts["maxMatsubara"]

    nlayers = len(layers)

    # Generate k-space grid
    k1s, k2s, _, _, area_factor = GKTH_find_radial_ks(
        p, deepcopy(layers), width=0.05, just_use_layer=layers_to_check
    )
    nrs, nangles = k1s.shape

    # Initialize Delta factors
    D_factors = np.zeros((nlayers, nrs, nangles))
    for i, L in enumerate(layers):
        L_copy = deepcopy(L)
        L_copy.Delta_0 = 1
        D_factors[i, :, :] = GKTH_Delta_k(p, L_copy, k1s, k2s)

    D_factors[np.isinf(D_factors) | np.isnan(D_factors)] = 0

    # Build the base Hamiltonian
    base_m = GKTH_hamiltonian_k(p, k1s, k2s, layers)

    # Matsubara calculations
    imaginary_identity_m = 1j * np.eye(4 * nlayers, 4 * nlayers)
    first_freqs = np.array([0, maxMatsubara])
    matsubara_freqs = np.zeros(maxCalcs)
    ksums = np.zeros((maxCalcs, nlayers))
    matsubara_freqs[:2] = first_freqs

    for itr in range(2):
        ksums[itr, :] = calculate_ksum(
            matsubara_freqs[itr],
            base_m,
            imaginary_identity_m,
            D_factors,
            area_factor,
            p,
            verbose,
        )

    # Iterative sampling
    prev_tol_check = 0
    stopped_changing = 0
    for itr in range(2, maxCalcs):
        # Compute differentials
        weights = ksums[:itr, layers_to_check[itr % len(layers_to_check)]]
        diffs = np.diff(weights, axis=0)
        if len(diffs) >= 2:
            ddiffs = np.gradient(diffs)
        else:
            ddiffs = 0
        matsdiffs = np.diff(matsubara_freqs[:itr])

        checks = matsdiffs > 1
        mats_idx = np.argmax(np.abs(checks * ddiffs * matsdiffs))
        new_n = (matsubara_freqs[mats_idx] + matsubara_freqs[mats_idx + 1]) // 2

        # Insert new frequency
        matsubara_freqs = np.insert(matsubara_freqs, mats_idx + 1, new_n)
        new_ksum = calculate_ksum(
            new_n, base_m, imaginary_identity_m, D_factors, area_factor, p, verbose
        )
        ksums = np.insert(ksums, mats_idx + 1, new_ksum, axis=0)

        # Convergence check
        if itr % 10 == 0:
            tol_check = []
            for i in range(nlayers):
                tol_check = trapezoid(
                    matsubara_freqs[:itr].reshape(-1, 1).repeat(nlayers, axis=1),
                    (np.abs(ksums[:itr, :]) + 0.5 * np.abs(ksums[0, :])),
                    axis=0,
                )
            tol_check = tol_check[layers_to_check]
            if (
                np.linalg.norm((tol_check - prev_tol_check) / prev_tol_check)
                < p.rel_tolerance_Greens
            ):
                stopped_changing += 1
                if stopped_changing == 2:
                    break
            else:
                stopped_changing = 0
            prev_tol_check = tol_check

    # Final integration
    Fs_sums = np.array(
        [
            trapezoid(matsubara_freqs[:itr], ksums[:itr, i]) + 0.5 * ksums[0, i]
            for i in range(nlayers)
        ]
    )

    # If verbose, return k-resolved function
    if verbose:
        F_kresolved_matsum = np.zeros((nlayers, nrs, nangles))
        for j in range(nlayers):
            for i1 in range(nrs):
                for i2 in range(nangles):
                    F_kresolved_matsum[j, i1, i2] = trapezoid(
                        matsubara_freqs[:itr], ksums[:itr, j]
                    )
        return Fs_sums, F_kresolved_matsum, k1s, k2s

    return Fs_sums, None, k1s, k2s


def GKTH_fix_lambda(p: GlobalParams, layer: Layer, Delta_target: float):
    """
    Finds and sets the BCS coupling constant to give a particular gap size
    at low temperature (default: 0.1 K).

    Parameters:
        p (GlobalParams): Global parameter object defining the stack and calculation parameters.
        layer (Layer): A single Layer object defining the superconducting layer.
        Delta_target (float): The target superconducting gap at 0.1 K.

    Returns:
        lambda_val (float): The calculated BCS coupling constant.
        layer (Layer): Updated layer object with the new BCS coupling constant.
    """

    if not isinstance(layer, Layer):
        raise ValueError("layer must be a single Layer object")

    layer.Delta_0 = Delta_target
    p.T = 0.1 * kB  # Set low temperature (0.1 K)
    p.h = 0  # No external magnetic field

    # Compute Green's function sum (Placeholder function: Implement separately)
    Fs_sums, _, _, _ = GKTH_Greens_radial(p, [layer])

    # Calculate lambda
    lambda_val = Delta_target / (p.T * abs(Fs_sums))[0]

    # Update layer's lambda value
    layer._lambda = lambda_val

    return lambda_val, layer
