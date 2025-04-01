import time
from typing import List

import numpy as np
from scipy.optimize import fmin

from GKTH_Greens_current_radial import GKTH_Greens_current_radial
from Global_Parameter import GlobalParams
from Layer import Layer


def GKTH_critical_current(p: GlobalParams, layers: List[Layer], **kwargs):
    """
    Finds the maximum current vs phase for a given structure

    Parameters:
    -----------
    p : object
        Global Parameter object defining the stack and calculation parameters.
    layers : list
        An array of Layer objects defining the stack

    Optional Parameters:
    -------------------
    layer_to_vary : int
        which layer to change the phase, default last
    initial_guess : float
        initial estimate of critical phase
    maxCalcs : int
        maximum calculations in current calculation
    spin_current : bool
        whether or not to return the spin-currents

    Returns:
    --------
    jc : float or array
        the critical current
    phase : float
        the critical phase
    """
    # Set default values for optional parameters
    layer_to_vary = kwargs.get("layer_to_vary", np.nan)
    initial_guess = kwargs.get("initial_guess", 1.5)
    maxCalcs = kwargs.get("maxCalcs", 500)
    spin_current = kwargs.get("spin_current", False)

    # If no layer to vary set, make it the last layer
    if np.isnan(layer_to_vary):
        layer_to_vary = len(layers) - 1  # Python uses 0-based indexing

    # The function to minimize to get jc. Returns -jc so a minimizer
    # can be used to find the maximum jc
    def jc_function(xs):
        start_time = time.time()
        if np.isscalar(xs):
            xs = [xs]
        js = np.zeros(len(xs))
        for i in range(len(xs)):
            x = xs[i]
            layers_temp = layers.copy()  # Create a copy of the layers
            layers_temp[layer_to_vary].phi = x
            j_t = GKTH_Greens_current_radial(p, layers_temp, maxCalcs=maxCalcs)
            js[i] = -j_t[0]
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        return js[0] if len(js) == 1 else js

    # The minimum search
    phase = fmin(jc_function, initial_guess, xtol=1e-3, ftol=1e10, disp=False)[0]
    phase = phase % (2 * np.pi)

    # Force the result between -pi and pi
    if phase > np.pi:
        phase = phase - 2 * np.pi

    if spin_current:
        layers[layer_to_vary].phi = phase
        jc = GKTH_Greens_current_radial(p, layers, maxCalcs=maxCalcs)
    else:
        jc = -jc_function(phase)

    return jc, phase
