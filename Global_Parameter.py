import numpy as np
from dataclasses import dataclass, field

@dataclass
class GlobalParams:
    """
    Description
    The Global_Params object contains paramters which apply to a stack of
    Layers like temperature, applied field and tunnelling paramters, as 
    well as paramters for controlling the calculation such as function 
    tolerances and whether to use symmetries or subsampling.
    """
    # Constants
    kB: float = 8.617333262145e-5  # Boltzmann constant in eV/K
    
    # Properties
    T: float = 5 * kB   # this is 5 K
    h: float = 0        # Global applied field strength
    theta: float = 0    # Angle of applied field about y, away from quantization axis z
    theta_ip: float = 0 # Angle of applied field about z, the quantization axis
    
    # Tunnelling parameters. Should be length nlayers-1.
    ts: np.ndarray = field(default_factory=lambda: np.zeros(100) + 0.9)
    cyclic_tunnelling: bool = True     # Whether the last layer tunnels to the first layer
    
    a: float = 3.30e-10  # lattice parameter
    
    interface_normal: int = 3       # direction of broken inversion symmetry. x=1,y=2,z=3
    
    # How big to make the area in k-space. 1 goes to 1st Brillouin
    # zone, 2 goes to the 3rd Brillouin zone, 3 goes to the 6th
    # Brillouin zone.
    BZ_multiplier: int = 1

    # Number of kpoints: must be multiple of 8 for meshing to work with mm symmetry
    # Minimum 80 for testing, 160 or greater for real use
    nkpoints: int = 80
    m_symmetry_line: float = 0
    lattice_symmetry: str = '4mm'
    use_kspace_subsampling: bool = True
    use_4mm_symmetry= True

    # The fraction of the number of points to aim for when subsampling. 1 is no subsampling, 
    # 0.1 is 10% the number of points. Default 0.1.
    subsampling_point_fraction: float = 0.1
    use_matsubara_subsampling: bool = True
    nradials: int = 50
    ntest: int = 2000
    nfinal: int = 200
    abs_tolerance_Greens: float = 1e-99
    rel_tolerance_Greens: float = 1e-5
    abs_tolerance_hc: float = 1e-8
    abs_tolerance_self_consistency_1S: float = 1e-6
    abs_tolerance_self_consistency_iterate: float = 1e-7
    rel_tolerance_self_consistency_iterate: float = 1e-6

    def __post_init__(self):
        self.nkpoints = self.adjust_nkpoints(self.nkpoints)

    @staticmethod
    def adjust_nkpoints(value):
        value = int(value)
        return value + 15 - (value - 1) % 16
    
    # k1 and k2 are nxn arrays, defined by obj.nkpoints, and contains 
    # k1 or k2 values at each point. The diagonals are shifted slightly 
    # to avoid divide-by-zeros when k1=k2.
    @property
    def k1(self):
        scale = self.BZ_multiplier * np.pi / self.a
        if self.lattice_symmetry in ["mm", "4mm"]:
            ks = scale * np.linspace((1 / (2 * self.nkpoints)), (1 - 1 / (2 * self.nkpoints)), self.nkpoints)
            k1, _ = np.meshgrid(ks, ks)
            for i in range(self.nkpoints):
                k1[i, i] = k1[i, i] + (2 * (i % 2) - 1) * scale / (self.nkpoints * 10)  # shift by a 10th of a step size with alternating sign to avoid divide by zeros
        else:
            ks = scale * np.linspace((-1 + 1 / (self.nkpoints)), (1 - 1 / (self.nkpoints)), self.nkpoints)
            k1, _ = np.meshgrid(ks, ks)
            for i in range(self.nkpoints):
                k1[i, i] = k1[i, i] + (2 * (i % 2) - 1) * scale / (self.nkpoints * 10)  # shift by a 10th of a step size with alternating sign
                k1[i, self.nkpoints - i - 1] = k1[i, self.nkpoints - i - 1] + (2 * (i % 2) - 1) * scale / (self.nkpoints * 10)  # shift by a 5th of a step size with alternating sign
        return k1

    @property
    def k2(self):
        scale = self.BZ_multiplier * np.pi / self.a
        if self.lattice_symmetry in ["mm", "4mm"]:
            ks = scale * np.linspace((1 / (2 * self.nkpoints)), (1 - 1 / (2 * self.nkpoints)), self.nkpoints)
            _, k2 = np.meshgrid(ks, ks)
            for i in range(self.nkpoints):
                k2[i, i] = k2[i, i] - (2 * (i % 2) - 1) * scale / (self.nkpoints * 10)  # shift by a 10th of a step size with alternating sign
        else:
            ks = scale * np.linspace((-1 + 1 / (self.nkpoints)), (1 - 1 / (self.nkpoints)), self.nkpoints)
            _, k2 = np.meshgrid(ks, ks)
            for i in range(self.nkpoints):
                k2[i, i] = k2[i, i] - (2 * (i % 2) - 1) * scale / (self.nkpoints * 10)  # shift by a 10th of a step size with alternating sign
                # Adjusting index for Python's 0-based index system
                k2[i, self.nkpoints - i - 1] = k2[i, self.nkpoints - i - 1] + (2 * (i % 2) - 1) * scale / (self.nkpoints * 10)  # shift by a 10th of a step size with alternating sign

        return k2

    @property
    def k_step_size(self):
        if self.lattice_symmetry in ["mm", "4mm"]:
            return self.BZ_multiplier * np.pi / (self.a * self.nkpoints)
        else:
            return 2 * self.BZ_multiplier * np.pi / (self.a * self.nkpoints)
