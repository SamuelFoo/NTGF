import numpy as np

def rotate_y(m, theta):
    Ry = np.array([
        [np.cos(theta / 2), np.sin(theta / 2)],
        [-np.sin(theta / 2), np.cos(theta / 2)]
    ])
    Ryt = Ry.T
    return Ry @ m @ Ryt

def rotate_z(m, theta_ip):
    Rz = np.array([
        [np.exp(1j * theta_ip / 2), 0],
        [0, np.exp(-1j * theta_ip / 2)]
    ])
    Rzt = Rz.T
    return Rz @ m @ Rzt

def GKTH_hamiltonian(p, layers):
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
        m[idx:idx+2, idx:idx+2, :, :] = xis + h_layer + h_global + SOC
        m[idx+2:idx+4, idx+2:idx+4, :, :] = -np.conj(xis + h_layer + h_global + SOC)

        # Superconductivity matrix
        m[idx, idx+3, :, :] = np.exp(1j * L.phi) * L.Ds(p)
        m[idx+1, idx+2, :, :] = -np.exp(1j * L.phi) * L.Ds(p)
        m[idx+2, idx+1, :, :] = -np.exp(-1j * L.phi) * L.Ds(p)
        m[idx+3, idx, :, :] = np.exp(-1j * L.phi) * L.Ds(p)

    # Tunneling between layers
    signs = [-1, -1, 1, 1]
    for i in range(nlayers - 1):
        idx = (i * 4)
        for j in range(4):
            m[idx+j, idx+4+j, :, :] = p.ts[i] * signs[j]
            m[idx+4+j, idx+j, :, :] = p.ts[i] * signs[j]

    if p.cyclic_tunnelling:
        shift = 4 * (nlayers - 1)
        for j in range(4):
            m[j, j+shift, :, :] = p.ts[i] * signs[j]
            m[j+shift, j, :, :] = p.ts[i] * signs[j]
    return m
