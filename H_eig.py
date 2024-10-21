import numpy as np
from Hamiltonian import GKTH_hamiltonian

def GKTH_find_spectrum(p, layers):
    """
    Description:
    Finds the eigenvalues of a layer stack, which correspond to the band energies.
    
    Inputs:
        p      : Global Parameter object defining the stack and calculation parameters.
        layers : A list of Layer objects defining the stack.
    
    Outputs:
        eigenvalues: An nkpoints x nkpoints x 4*nlayers array of k-resolved eigenvalues.
    """
    # Set all Deltas to zero and get Hamiltonian
    nlayers = len(layers)

    Hs = GKTH_hamiltonian(p,layers);
    #Hs is 4*nlayers x 4*nlayers x nkpoints x nkpoints
    # Calculate eigenvalues for each point in the k-space grid
    nkpoints = p.nkpoints
    eigenvalues = np.zeros((nkpoints, nkpoints, 4 * nlayers))
    for i in range(nkpoints):
        #print(f"Hamiltonian at i={i}")
        for j in range(nkpoints):
            try:
                eigenvalues[i, j, :] = np.linalg.eigvals(Hs[:, :, i, j])
            except:
                eigenvalues[i, j, :] = np.nan
        #print(f"Hamiltonian at i={i}")
    return eigenvalues

