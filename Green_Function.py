import numpy as np
from Hamiltonian import GKTH_hamiltonian
from Delta import GKTH_Delta
from ksubsample import GKTH_ksubsample
from k_space_flipping import GKTH_flipflip
from scipy.linalg import inv
import matplotlib.pyplot as plt

def GKTH_Greens(p, layers, density_grid=None, compute_grid=None, maxCalcs=500, maxMatsubara=1e7, verbose=False):
    """
    GKTH_Greens function to compute the anomalous Green's function over k and Matsubara frequencies.

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
    #print("Getting GF and delta is ", layers[0].Delta_0)
    nlayers = len(layers)
    #print(layers[0].Delta_0)
    D_factors = np.zeros((p.nkpoints, p.nkpoints, nlayers))
    
    for i in range(nlayers):
        D_factors[:,:,i] = 1.0 / GKTH_Delta(p, layers[i], 1)
    D_factors[np.isnan(D_factors)] = 0

    normalisation_factor = p.k_step_size**2 / (2 * np.pi / p.a)**2

    # Base matrices with no Matsubara frequency dependence
    base_m = GKTH_hamiltonian(p, layers)
    imaginary_identity_m = 1j * np.eye(4 * nlayers)

    # Create matrices for k-space subsampling if not already provided
    if density_grid is None or density_grid.shape != (p.nkpoints, p.nkpoints):
        density_grid, compute_grid = GKTH_ksubsample(p, layers)
    
    # Flatten the 2D arrays for linear indexing
    compute_idxs = np.transpose(np.nonzero(compute_grid == 1))
    npointscalc = len(compute_idxs)
    random_sampling_max = density_grid[compute_idxs[:,0], compute_idxs[:,1]] - 1

    # Symmetry multiplier
    if p.lattice_symmetry == '4mm':
        symmetry_multiplier = 8 * np.ones((p.nkpoints, p.nkpoints))
        np.fill_diagonal(symmetry_multiplier, 4)
    elif p.lattice_symmetry == 'mm':
        symmetry_multiplier = 4 * np.ones((p.nkpoints, p.nkpoints))
    elif p.lattice_symmetry == 'm':
        symmetry_multiplier = 2 * np.ones((p.nkpoints, p.nkpoints))
    else:
        symmetry_multiplier = np.ones((p.nkpoints, p.nkpoints))
    
    overall_multiplier = symmetry_multiplier * density_grid**2 * compute_grid
    overall_multiplier_flat = overall_multiplier[compute_idxs[:,0], compute_idxs[:,1]]

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

    matsubara_freqs_unsrt = np.zeros(maxCalcs)
    F_kresolved = np.zeros((maxCalcs, nlayers, npointscalc))
    F_kresolved_symm = np.zeros((nlayers, p.nkpoints, p.nkpoints))
    F_kresolved_final = np.zeros((nlayers, 2 * p.nkpoints, 2 * p.nkpoints))

    # Calculate ksums for the first frequencies
    matsubara_freqs[:L] = first_freqs
    for itr in range(L):
        ksums[itr, :] = calculate_ksum(matsubara_freqs[itr], nlayers, npointscalc, base_m, imaginary_identity_m, 
                                       compute_idxs, D_factors, overall_multiplier_flat, random_sampling_max, p, 
                                       verbose, normalisation_factor)

        if verbose:
            values[itr, :] = np.concatenate(([matsubara_freqs[itr]], np.abs(ksums[itr, :])))
    weights[:L] = np.sum(np.abs(ksums[:L, :]), axis=1)

    # Iterate and dynamically sample new points based on weighting
    prev_tol_check = 0
    stopped_changing = 0
    for itr in range(L, maxCalcs):
        # Calculate all the differentials
        diffs[:itr - 1] = np.diff(weights[:itr])
        # Calculate the second-order gradient only when there are enough points
        if itr > 2:
            if len(diffs[:itr - 2]) >= 2:  # Ensure there are at least 2 elements
                ddiffs[:itr - 2] = np.gradient(diffs[:itr - 2])
            else:
                # Handle small cases manually or set a default value
                ddiffs[:itr - 2] = 0  # or some other default behavior
        elif itr == 2:
            ddiffs[0] = diffs[1] - diffs[0]  # Simple difference for 2 points

        matsdiffs[:itr - 1] = np.diff(matsubara_freqs[:itr])

        # Exclude points next to each other
        checks = matsdiffs[:itr - 1] > 1

        # Find the new Matsubara frequency index
        mats_idx = np.argmax(np.abs(checks * ddiffs[:itr - 1] * matsdiffs[:itr - 1]))
        new_n = (matsubara_freqs[mats_idx] + matsubara_freqs[mats_idx + 1]) // 2

        # Insert the new frequency
        matsubara_freqs = np.insert(matsubara_freqs, mats_idx + 1, new_n)

        new_ksum = calculate_ksum(new_n, nlayers, npointscalc, base_m, imaginary_identity_m, compute_idxs, 
                                  D_factors, overall_multiplier_flat, random_sampling_max, p, verbose, normalisation_factor)
        ksums = np.insert(ksums, mats_idx + 1, new_ksum, axis=0)
        weights = np.insert(weights, mats_idx + 1, np.sum(np.abs(new_ksum)))

        # Verbose tracking
        if verbose:
            integrals[itr - L] = np.trapz(matsubara_freqs[:itr], weights[:itr]) + 0.5 * weights[0]
            values[itr, :] = np.concatenate(([new_n], np.abs(new_ksum)))

        # Convergence check every 10 iterations
        if itr % 10 == 0:
            tol_check = np.trapz(matsubara_freqs[:itr], weights[:itr]) + 0.5 * weights[0]
            if np.abs((tol_check - prev_tol_check) / prev_tol_check) < p.rel_tolerance_Greens:
                stopped_changing += 1
                if stopped_changing == 3:
                    break
            else:
                stopped_changing = 0
            prev_tol_check = tol_check

    # Final integration
    Fs_sums = np.zeros(nlayers)
    for layer in range(nlayers):
        Fs_sums[layer] = np.trapz(matsubara_freqs[:itr], ksums[:itr, layer]) + 0.5 * ksums[0, layer]

    if verbose:
        id_sorted = np.argsort(matsubara_freqs_unsrt[:itr])  # Capture only the indices
        F_kresolved = F_kresolved[id_sorted, :, :]
        for j in range(nlayers):
            for i in range(npointscalc):
                F_kresolved_symm[j, compute_idxs[i, 0], compute_idxs[i, 1]] = (
                    np.trapz(matsubara_freqs[:itr], F_kresolved[:itr, j, i]) + 
                    0.5 * F_kresolved[0, j, i]
                )
            F_kresolved_final[j, :, :] = GKTH_flipflip(
                F_kresolved_symm[j, :, :], 
                use_4mm_symmetry = p.use_4mm_symmetry, 
                use_kspace_subsampling = p.use_kspace_subsampling, 
                density_grid = density_grid
            )
        plt.figure(figsize=(12, 6))  # Define the figure size
        
        # First subplot: pcolor plot of real(F_kresolved_final)
        plt.subplot(1, 2, 1)
        plt.pcolormesh(np.real(np.squeeze(F_kresolved_final[0, :, :])), cmap='viridis')
        plt.colorbar()  # Optional: to show a colorbar
        plt.title('Real part of F_kresolved_final')
        
        # Second subplot: plot of Matsubara frequencies vs abs(ksums)
        plt.subplot(1, 2, 2)
        plt.plot(matsubara_freqs, np.abs(ksums), 'o-')
        plt.xlim(0,5200)
        plt.title('Matsubara Frequencies vs |ksums|')
        plt.xlabel('Matsubara Frequencies')
        plt.ylabel('|ksums|')
        
        # Display the plots
        plt.tight_layout()

    return Fs_sums, matsubara_freqs
    #return Fs_sums, matsubara_freqs, ksums, integrals, values, F_kresolved_final

def calculate_ksum(n, nlayers, npointscalc, base_m, imaginary_identity_m, compute_idxs, 
                   D_factors, overall_multiplier_flat, random_sampling_max, p, verbose, normalisation_factor):
    Fs_ksum = np.zeros(nlayers)
    w = (2 * n + 1) * np.pi * p.T
    ws = imaginary_identity_m * w
    Fupdowns = np.zeros((nlayers, npointscalc))
    Fdownups = np.zeros((nlayers, npointscalc))

    idx_samps = compute_idxs[:, 0] + np.round(random_sampling_max * np.random.rand(npointscalc)).astype(int) 
    + p.nkpoints * np.round(random_sampling_max * np.random.rand(npointscalc)).astype(int)
    
    for i in range(npointscalc):
        idx_samp = idx_samps[i]
        m_inv = np.linalg.inv(-base_m[:, :, compute_idxs[i, 0], compute_idxs[i, 1]] + ws)
        for j in range(nlayers):
            # Extract Fupdown and Fdownup for each layer
            #print(m_inv[4 * j, 4 * j + 3], m_inv[4 * j + 1, 4 * j + 2])
            Fupdowns[j, i] = m_inv[4 * j, 4 * j + 3] * D_factors[compute_idxs[i, 0], compute_idxs[i, 1], j]
            Fdownups[j, i] = m_inv[4 * j + 1, 4 * j + 2] * D_factors[compute_idxs[i, 0], compute_idxs[i, 1], j]
            
    # Calculate the k-space sum over all layers
    for layer in range(nlayers):
        Fs_ksum[layer] = normalisation_factor * np.sum(overall_multiplier_flat * (Fupdowns[layer, :] - Fdownups[layer, :]))

        # Store verbose data if required
        if verbose:
            F_kresolved = Fupdowns[layer, :] - Fdownups[layer, :]
            matsubara_freqs_unsrt = n  # Track the current Matsubara frequency for later sorting
    #print(Fs_ksum)
    return Fs_ksum

"""def calculate_ksum(n, nlayers, npointscalc, base_m, imaginary_identity_m, compute_idxs,
                   D_factors, overall_multiplier_flat, random_sampling_max, p, verbose, normalisation_factor):
    # Initialize the sum for each layer
    Fs_ksum = np.zeros(nlayers)
    # Compute the Matsubara frequency
    w = (2 * n + 1) * np.pi * p.T
    ws = imaginary_identity_m * w
    # Allocate arrays for Fupdown and Fdownup
    Fupdowns = np.zeros((nlayers, npointscalc))
    Fdownups = np.zeros((nlayers, npointscalc))

    # Generate the random sampling of k-points using random_sampling_max
    idx_samps = compute_idxs[:, 0] + np.round(random_sampling_max * np.random.rand(npointscalc)).astype(int) \
                + p.nkpoints * np.round(random_sampling_max * np.random.rand(npointscalc)).astype(int)
    # Ensure idx_samps stays within bounds of base_m's third axis size
    idx_samps = np.mod(idx_samps, base_m.shape[2])
    # Loop over all sampled k-points
    for i in range(npointscalc):
        idx_samp = idx_samps[i]
        # Invert the Hamiltonian matrix for the given k-point and Matsubara frequency
        m_inv = np.linalg.inv(base_m[:, :, compute_idxs[i, 0], compute_idxs[i, 1]] + ws)
        for j in range(nlayers):
            # Extract Fupdown and Fdownup for each layer
            print(m_inv[4 * j, 4 * j + 3], m_inv[4 * j + 1, 4 * j + 2])
            Fupdowns[j, i] = m_inv[4 * j, 4 * j + 3] * D_factors[compute_idxs[i, 0], compute_idxs[i, 1], j]
            Fdownups[j, i] = m_inv[4 * j + 1, 4 * j + 2] * D_factors[compute_idxs[i, 0], compute_idxs[i, 1], j]
            
    # Calculate the k-space sum over all layers
    for layer in range(nlayers):
        Fs_ksum[layer] = normalisation_factor * np.sum(overall_multiplier_flat * (Fupdowns[layer, :] - Fdownups[layer, :]))

        # Store verbose data if required
        if verbose:
            F_kresolved = Fupdowns[layer, :] - Fdownups[layer, :]
            matsubara_freqs_unsrt = n  # Track the current Matsubara frequency for later sorting

    return Fs_ksum"""