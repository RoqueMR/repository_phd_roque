import numpy as np
import eos.bsk.fits_bsk as fb

n = np.array([0.1, 0.2])


BSk22 = fb.load_eos("BSk22")

print(BSk22.pressure_equilibrium(n))
