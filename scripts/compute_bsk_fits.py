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

print(type(BSk22.load_p_i_coefficients("TableC1")))
print(type(np.array(8.0)))

print(cc.alpha *  1.0)
