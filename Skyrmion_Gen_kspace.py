# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 01:11:51 2024
Plot skyrmion in real and kspace
@author: hp
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from numpy import pi, inf
from scipy.optimize import minimize, minimize_scalar

def skyrm1_Mxyi(kx, ky, c, w, p, v, m):
    """
    This function creates the spin vector for a given spin at position kx, ky
    defined from the origin (0,0), 
    c is the center of the DW, w is the width of the DW,
    p is the polarity, 
    v is the angle at the DW or chirality, 
    m is the vorticity or number of cycles. 
    ThetaR is the polar angle of spin at a position given the distance 
    from the center of the skyrmion, 
    phi is the azimuthal angle.    
    equation taken from : DOI: 10.1038/ncomms13613
    """
    r = np.sqrt(kx**2 + ky**2)
    phi3 = np.arctan2(kx + 1.23e-9, ky + 1.23e-9) + pi
    ThetaR = m * (np.arcsin(np.tanh((r + c) / (w + 1.23e-9))) + np.arcsin(np.tanh((r - c) / (w + 1.23e-9))))
    Mi = [np.cos(m * phi3 + v) * np.sin(ThetaR), np.sin(m * phi3 + v) * np.sin(ThetaR), p * np.cos(ThetaR)]
    return Mi

def Calc_Skyrm_size(A, K, D):
    """
    This function calculates the radius and DW width of a skyrmion,
    by taking the magnetic interaction constants, 
    A- exchange (pJ/m),
    D- DMI and (mJ/m^2),
    K - anisotropy (MJ/m^3) as inputs
    equation taken from : DOI: 10.1038/s42005-018-0029-0
    """
    A = A * 1e-12
    K = K * 1e6
    D = D * 1e-3
    R = pi * D * np.sqrt(A / (16 * A * (K**2) - (pi**2) * (K * D**2)))
    W = (pi * D) / (4 * K)
    return R, W

def Skyrm_energy(params, args):
    """
    This function calculates the radius and DW width of a skyrmion,
    by taking the magnetic interaction constants, 
    A- exchange (pJ/m),
    D- DMI and (mJ/m^2),
    K - anisotropy (MJ/m^3) as inputs
    equation taken from : DOI: 10.1038/s42005-018-0029-0
    """
    R = params[0] * 1e-9
    B = args[0] * 1e-3 # mT to T
    Ms = args[1] * 1e3 # kA/m to A/m
    t = args[2] * 1e-9 # nm to m
    A = args[3] * 1e-12 # pJ/m to J/m 
    D = args[4] * 1e-3 # mJ/m^2 to J/m^2
    K = args[5] * 1e6 # MJ/m^3 to J/m^3
    w = (pi * D) / (4 * K)
    E = 4 * pi * t * (A * (R / w + w / R) - pi * D * R + K * w * R + Ms * B * (0.5 * R**2 + (pi**2) * (w**2)) / 24)
    return abs(E)

t = 0.4 # thickness in nm
Ms = 960 # kA/m
B = 25 # mT
A = 10 # pJ/m
K = 0.0228 # J/m^3
D = 0.7 # mJ/m^2

mu0 = 0.0000012566371
res = minimize(Skyrm_energy, x0=[1201, 37], args=[B, Ms, t, A, D, K], bounds=((0, 1.537e-6), (0, 1.537e-6)))
# Energy based skyrmion radius is not used right now, not working

c_dw = 50 # distance of dw from center of skyrmion
w_dw = 20 #width of the domain wall 
print(res)
print(['c_dw', c_dw * 1e9, 'w_dw', w_dw * 1e9])

Q = 1 # topological number 
phi = 0#pi / 2 # Azimuthal angle of DW - chirality 

ni_gridsize = 250 # always have this as an odd number to include 0
kx = ky = np.linspace(-(ni_gridsize - 1) / 2, +(ni_gridsize - 1) / 2, ni_gridsize, dtype=float)
nx = ny = np.linspace(0, ni_gridsize, ni_gridsize, dtype=int)

kxp, kyp = np.meshgrid(kx, ky, indexing='ij')
nxp, nyp = np.meshgrid(kx, ky, indexing='ij')

Mx, My, Mz = skyrm1_Mxyi(kxp, kyp, c_dw, w_dw, 1, phi, Q)

# Perform 2D Fourier transform on each component
Mx_k = np.fft.fftshift(np.fft.fft2(Mx))
My_k = np.fft.fftshift(np.fft.fft2(My))
Mz_k = np.fft.fftshift(np.fft.fft2(Mz))

# Plotting Mx_k, My_k, Mz_k in k-space
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8), dpi=80, facecolor='w', edgecolor='k')
plt.rc('lines', linewidth=4)

#The log function is used to enhance visibility of low amplitude components and reduce large amplitudes
#it can be removed per se.
# ax1.imshow(np.log1p(np.abs(Mx_k)), cmap='viridis', extent=(-100, 100, -100, 100))
ax1.imshow(np.abs(Mx_k), cmap='viridis', extent=(-100, 100, -100, 100))
ax1.set_xlabel('kx', fontsize=18)
ax1.set_ylabel('ky', fontsize=18)
ax1.set_title('Mx_k')

# ax2.imshow(np.log1p(np.abs(My_k)), cmap='viridis', extent=(-100, 100, -100, 100))
ax2.imshow(np.abs(My_k), cmap='viridis', extent=(-100, 100, -100, 100))
ax2.set_xlabel('kx', fontsize=18)
ax2.set_ylabel('ky', fontsize=18)
ax2.set_title('My_k')

# ax3.imshow(np.log1p(np.abs(Mz_k)), cmap='viridis', extent=(-100, 100, -100, 100))
ax3.imshow(np.abs(Mz_k), cmap='viridis', extent=(-100, 100, -100, 100))
ax3.set_xlabel('kx', fontsize=18)
ax3.set_ylabel('ky', fontsize=18)
ax3.set_title('Mz_k')

plt.show()

# M_rgb = np.ones((Lx, Ly, 3), np.double)
# M_rgb2 = M_rgb + 1
# M_rgb[:, :, 0] = (M[0] + 1) / 2 * M_rgb[:, :, 0]
# M_rgb[:, :, 1] = (M[1] + 1) / 2 * M_rgb[:, :, 1]
# M_rgb[:, :, 2] = (M[2] + 1) / 2 * M_rgb[:, :, 2]

m_thresh = 0.1
Mx = Mx * (np.abs(Mx) >= m_thresh)
My = My * (np.abs(My) >= m_thresh)
Mz = Mz * (np.abs(Mz) >= m_thresh)

# for i in range(len(org)):
#     xinds = np.asarray(nxp + org[i][0], int)
#     yinds = np.asarray(nyp + org[i][1], int)
#     M_rgb[xinds, yinds, 0] = (Mx + 1) / 2
#     M_rgb[xinds, yinds, 1] = (My + 1) / 2
#     M_rgb[xinds, yinds, 2] = 0.5 * (Mz + 1) / 2

# M_show = (M_rgb * 255).astype(np.uint8)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8), dpi=80, facecolor='w', edgecolor='k')
plt.rc('lines', linewidth=4)

fig.tight_layout(pad=3)

ax1.imshow(Mx, cmap='RdGy')
ax1.set_xlabel('x', fontsize=18)
ax1.set_ylabel('y', fontsize=18)
ax1.set_title('Mx')

ax2.imshow(My, cmap='RdGy')
ax2.set_xlabel('x', fontsize=18)
ax2.set_ylabel('y', fontsize=18)
ax2.set_title('My')

ax3.imshow(Mz, cmap='RdGy')
ax3.set_xlabel('x', fontsize=18)
ax3.set_ylabel('y', fontsize=18)
ax3.set_title('Mz')

plt.show()