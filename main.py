from Global_Parameter import GlobalParams
from Layer import Layer
from H_eig import GKTH_find_spectrum
import matplotlib.pyplot as plt
import numpy as np
from H_eig import GKTH_find_spectrum
from k_space_flipping import GKTH_flipflip
from Green_Function import GKTH_Greens
from self_consistency_delta import GKTH_self_consistency_1S

p = GlobalParams()
layers = [Layer()]

delta,_, _ = GKTH_self_consistency_1S(p,layers)

# Assuming GKTH_find_spectrum and GKTH_flipflip are defined properly
kx = np.array(p.k1)
ky = np.array(p.k2)
max_val_kx = np.max(np.abs(kx))
kx = np.linspace(-max_val_kx, max_val_kx, p.nkpoints*2)
ky = np.linspace(-max_val_kx, max_val_kx, p.nkpoints*2)
# Ensure kx and ky are properly ranged and meshed
kx, ky = np.meshgrid(kx, ky)

energy = GKTH_find_spectrum(p, layers)

# energy is 3d array, nkpoints x nkpoints x (4*nlayers)
# each k-point has its only eigenvalues list

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(4*len(layers)):
    energy_band = energy[:, :, i]
    new_band = GKTH_flipflip(energy_band)
    surf = ax.plot_surface(kx, ky, new_band, cmap='viridis', alpha=0.5)

fig.colorbar(surf)
#ticks = np.linspace(-max_val_kx, max_val_kx, 5)  # Adjust the number of ticks as needed
#ax.set_xticks(ticks)
#ax.set_yticks(ticks)

ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.set_zlabel('Energy')
ax.set_title('Energy Spectrum')

#print("gap is ",delta)
plt.show()

