from Global_Parameter import GlobalParams
from Layer import Layer
from H_eig import GKTH_find_spectrum
import matplotlib.pyplot as plt
import numpy as np
from H_eig import GKTH_find_spectrum
from k_space_flipping import GKTH_flipflip
from Green_Function import GKTH_Greens
from self_consistency_delta import GKTH_self_consistency_1S
from dataclasses import field

p = GlobalParams()
Nb = Layer()
layers = [Nb]

"""# Assuming GKTH_find_spectrum and GKTH_flipflip are defined properly
kx = np.array(p.k1)
ky = np.array(p.k2)
max_val_kx = np.max(np.abs(kx))
kx = np.linspace(-max_val_kx, max_val_kx, p.nkpoints*2)
ky = np.linspace(-max_val_kx, max_val_kx, p.nkpoints*2)
# Ensure kx and ky are properly ranged and meshed
kx, ky = np.meshgrid(kx, ky)"""

"""Ts=[1,4,7]
hs = [0.0001,0.0002]
T_init = p.T
fig = plt.figure()
for T in Ts:
    p.T = T_init*T
    gap_array = []
    for i in range(len(hs)):
        print("T and h are",T, hs[i])
        p.h = hs[i]
        gap_array.append(GKTH_self_consistency_1S(p,layers)[0])
    plt.scatter(hs,gap_array, label = "T = "+str(T)+" K")
plt.legend()
plt.show()"""




# energy is 3d array, nkpoints x nkpoints x (4*nlayers)
# each k-point has its only eigenvalues list
"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define a color list (using some common colors, adjust as needed)
color_list = ['blue', 'green', 'cyan', 'yellow', 'purple', 'orange', 'brown', 'red', 'magenta']

for i in range(4*len(layers)):
    energy_band = energy[:, :, i]
    print(f"{energy[0,0,i]:.3}")
    new_band = GKTH_flipflip(energy_band)
    # Check for NaNs or infinities before plotting
    if np.isnan(new_band).any() or np.isinf(new_band).any():
        print(f"Skipping band {i} due to NaN or infinity values.")
        continue
    # Set the color, cycling through color_list
    color = color_list[i % len(color_list)]
    ax.scatter(kx, ky, new_band, color = color, alpha=0.1, s=1)"""
    #surf = ax.plot_surface(kx, ky, new_band, cmap='viridis', alpha=0.5)
p.h = 0.001
p.T = p.T*4
Delta, layers, residual_history = GKTH_self_consistency_1S(p, layers)

"""ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.set_zlabel('Energy (eV)')
ax.set_title('Energy Spectrum')

#print("gap is ",delta)
plt.show()"""

