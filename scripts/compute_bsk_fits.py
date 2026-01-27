import numpy as np
import eos.bsk.fits_bsk as fb
from astropy import constants as cc
from astropy import units as u
import matplotlib.pyplot as plt


m_mu = 1.883531627e-28 * u.kg

n = np.loadtxt("/home/roque/Desktop/nb.txt")


BSk22 = fb.load_eos("BSk22")
BSk24 = fb.load_eos("BSk24")
BSk25 = fb.load_eos("BSk25")
BSk26 = fb.load_eos("BSk26")

print(BSk22.total_deriv_pressure_core(n))
print(BSk24.total_deriv_pressure_core(n))
print(BSk25.total_deriv_pressure_core(n))
print(BSk26.total_deriv_pressure_core(n))
