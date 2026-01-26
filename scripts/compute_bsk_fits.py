import numpy as np
import eos.bsk.fits_bsk as fb
from astropy import constants as cc
from astropy import units as u
import matplotlib.pyplot as plt



n = np.loadtxt("/home/roque/Desktop/nb.txt") 


BSk22 = fb.load_eos("BSk22")
BSk24 = fb.load_eos("BSk24")
BSk25 = fb.load_eos("BSk25")
BSk26 = fb.load_eos("BSk26")


n_e = BSk22.e_num_dens_core(n)

print(BSk22.deriv_pressure_equilibrium(n))
