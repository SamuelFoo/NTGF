import numpy as np


def GKTH_flipflip(
    m, use_4mm_symmetry=True, use_kspace_subsampling=True, density_grid=None
):
    """
    GKTH_flipflip constructs a full matrix out of a sub-calculated matrix
    from the symmetry used. E.g., the new Fs sum is often calculated only on
    1/8 of the Brillouin zone if 4mm symmetry is used. Flip flip reconstructs
    the rest of the Brillouin zone from this matrix.

    Args:
    m (np.array): The input matrix to reconstruct.
    use_4mm_symmetry (bool): Whether 4mm symmetry was used to construct the matrix m.
    use_kspace_subsampling (bool): Whether grid-based k-space subsampling was used
                                   in the calculation.
    density_grid (np.array): The density grid for k-space subsampling if used.

    Returns:
    np.array: The input matrix after reconstruction.
    """
    # Initialize density grid if not provided
    if density_grid is None:
        density_grid = np.ones(m.shape)

    len_m = len(m)

    # Apply k-space subsampling if enabled
    if use_kspace_subsampling:
        for i in range(len_m):
            for j in range(i, len_m):
                d = density_grid[i, j]
                m[i, j] = m[int((i) // d) * int(d), int((j) // d) * int(d)]

    # Apply 4mm symmetry if enabled
    if use_4mm_symmetry:
        for i in range(len_m):
            for j in range(i, len_m):
                m[j, i] = m[i, j]

    # Create a new matrix to hold the reconstructed full matrix
    new_m = np.zeros((2 * len_m, 2 * len_m))

    # Assign the sub-calculated matrix and its flips to reconstruct the full matrix
    new_m[len_m:, len_m:] = m  # Bottom-right quadrant
    new_m[:len_m, len_m:] = np.flip(m, 0)  # Top-right quadrant, vertical flip
    new_m[len_m:, :len_m] = np.flip(m, 1)  # Bottom-left quadrant, horizontal flip
    new_m[:len_m, :len_m] = np.flip(np.flip(m, 0), 1)  # Top-left quadrant, both flips

    return new_m
