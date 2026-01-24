import yaml
import numpy as np
from astropy import constants as cc
from astropy import units as u
from fractions import Fraction
from pathlib import Path
from dataclasses import dataclass


# Resolve the directory where this script and the .yaml file are
this_files_path = Path(__file__).resolve().parent
# BSk .yaml file path
bsk_yaml_path = this_files_path / "parameters_bsk.yaml"
# Open .yaml path in reading mode
with open(bsk_yaml_path) as f:
    bsk_params_raw = yaml.safe_load(f)  # Load .yaml file


@dataclass(frozen=True)
class BSkEOS:
    name: str
    params: dict

    # ---- analytical fits ----
    def load_p_i_coefficients(self, table_name):
        p = self.params[table_name]  # table_name is a header in the .yaml file
        p_values = (p[f"p{i}"] for i in range(1, len(p) + 1))
        return p_values

    def equil_energy_per_nucleon(self, n):
        """
        Equilibrium energy per nucleon.
        See eq. (C1) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n (float or numpy.ndarray): mean baryon number density [fm^-3]

        Returns
            (float or numpy.ndarray): Equilibrium energy per nucleon [MeV]
        """
        p_values = self.load_p_i_coefficients("TableC1")
        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14 = p_values

        def w1(n):
            return 1.0 / (1.0 + p9 * n)

        def w2(n):
            return 1.0 / (1.0 + (p13 * n)**(p14))
        e_gr = -9.1536  # MeV
        part1 = (p1 * n)**(7/6)
        part2 = 1.0 + np.sqrt(p2 * n)
        part3 = 1.0 + np.sqrt(p3 * n)
        part4 = 1.0 + np.sqrt(p4 * n)
        part5 = 1.0 + np.sqrt(p5 * n)
        part6 = p6 * n**p7 * (1.0 + p8 * n) * (1.0 - w1(n)) * w2(n)
        part7 = (p10 * n)**p11 * (1.0 - w2(n)) / (1.0 + p12*n)
        return (
            e_gr  # constant term
            + (part1 * part4 * w1(n)) / (part2 * part3 * part5)  # low denss.
            + part6  # moderate densities
            + part7  # high densities
            )
    
    def tot_mass_ener_dens(self, n):
        """
        Total mass energy density
        See eq. (4) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n (float or numpy.ndarray): mean baryon number density [fm^-3]

        Returns
            (float or numpy.ndarray): Total mass energy density [g / cm^3]
        """
        e_eq = self.equil_energy_per_nucleon(n) * u.MeV  # e_eq [MeV]
        m_n_MeV = (cc.m_n * cc.c**2).to(u.MeV)  # n mass [MeV]
        rho = (n * u.fm**-3) * (e_eq + m_n_MeV) / cc.c**2
        return rho.to_value(u.g / u.cm**3)



def load_eos(name: str) -> BSkEOS:
    return BSkEOS(name=name, params=bsk_params_raw[name])
