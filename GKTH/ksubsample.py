import numpy as np
from scipy.optimize import minimize_scalar

from .H_eig import GKTH_find_spectrum


def GKTH_ksubsample(p, layers):
    """
    Description:
    Creates matrices used for grid-based k-space subsampling.
    Currently it does this by assuming the contribution to the Greens
    function roughly follows 1/xi, i.e. the k1,k2 which are close to the
    Fermi level for the different layers in the structure. If there are lots
    of layers with different Fermi surfaces this will not be very effective.

    Inputs:
        p:      Global Parameter object defining the stack and calculation parameters.
        layers: An array of Layer objects, defining the junction structure.

    Outputs:
        density_grid: An nxn matrix, the value m in each element describes that it is part of an mxm block to calculate.
        compute_grid: An nxn matrix of booleans. The value is True if it is in the bottom left corner of a sub-block defined by density_grid.
    """
    block_sizes = [1, 2, 3, 4]  # Final blocks will be 2^block_sizes
    spread = 1  # Default value
    # 'spread' determines how spread out or focussed the subsampling should be.
    # Lower values e.g. 0.8 lead to more aggressive high sampling/low sampling
    # areas. High values e.g. 1.2 leads to larger regions of intermediate
    # sampling rate. I think it's probably best to leave it at 1.

    # Function for making and refining the compute and density grids
    def make_grid(initial_spectrum, scale):
        # Calculate density grid initial values
        density_grid_initial = np.round(
            np.log((scale * initial_spectrum) ** (1 / spread))
        )
        density_grid_initial[density_grid_initial < 0] = 0
        density_grid_initial[density_grid_initial > max(block_sizes)] = max(block_sizes)
        density_grid_initial = 2**density_grid_initial

        # Initialize density and compute grids
        density_grid = np.ones((p.nkpoints, p.nkpoints))
        compute_grid = np.ones((p.nkpoints, p.nkpoints))

        # Process grid in blocks
        for b in 2 ** np.array(block_sizes):
            for i in range(p.nkpoints // b):
                for j in range(p.nkpoints // b):
                    # Check if all values in the block are greater than or equal to b
                    if np.all(
                        density_grid_initial[i * b : (i + 1) * b, j * b : (j + 1) * b]
                        >= b
                    ):
                        density_grid[i * b : (i + 1) * b, j * b : (j + 1) * b] = b
                        compute_grid[i * b : (i + 1) * b, j * b : (j + 1) * b] = 0
                        compute_grid[i * b, j * b] = 1

        # Apply lattice symmetry adjustments
        if p.lattice_symmetry == "4mm":
            for i in range(1, p.nkpoints):
                for j in range(i):
                    compute_grid[i, j] = 0
        elif p.lattice_symmetry == "m":
            compute_grid[: p.nkpoints // 2, : p.nkpoints // 2] = 0
            compute_grid[p.nkpoints // 2 :, p.nkpoints // 2 :] = 0

        # Calculate the number of points
        npoints = np.sum(density_grid**-2)

        return npoints, density_grid, compute_grid

    nlayers = len(layers)
    eigenvalues = GKTH_find_spectrum(p, layers)  # Needs implementation

    if p.use_kspace_subsampling and not np.isnan(eigenvalues).any():
        # Compute the average spectrum
        avg_spectrum = np.prod(np.abs(eigenvalues), axis=2) ** (1 / (4 * nlayers))

        # Change scale until desired point fraction is achieved
        def anonymous_function(x):
            return (
                make_grid(avg_spectrum, x)[0] / (p.nkpoints**2)
                - p.subsampling_point_fraction
            )

        # Options for the 'bounded' method (only 'xatol' is valid here)
        options_bounded = {"xatol": 0.01, "disp": 0}
        # General options for methods where 'fatol' is valid (e.g., Brent)
        options_brent = {"xatol": 0.01, "fatol": 0.01, "disp": 0}

        # Try different intervals for finding the best scale
        try:
            # First attempt with 'bounded' method and original bounds
            best_scale = minimize_scalar(
                anonymous_function,
                bounds=(0, 100),
                method="bounded",
                options=options_bounded,
            )
        except Exception as e:
            try:
                # Second attempt with different bounds and 'bounded' method
                best_scale = minimize_scalar(
                    anonymous_function,
                    bounds=(-100, 1000),
                    method="bounded",
                    options=options_bounded,
                )
            except Exception as e:
                # Final fallback using 'brent' method where 'fatol' is valid
                best_scale = minimize_scalar(
                    anonymous_function, method="brent", options=options_brent
                )

        # Extract the best scale and use it to get the density and compute grid
        best_scale = best_scale.x
        _, density_grid, compute_grid = make_grid(avg_spectrum, best_scale)
    else:
        # If k-space subsampling is not used, create default grids
        density_grid = np.ones((p.nkpoints, p.nkpoints))
        compute_grid = np.ones((p.nkpoints, p.nkpoints))

    return density_grid, compute_grid
