import yaml
import numpy as np
from astropy import constants as cc
from astropy import units as u
from fractions import Fraction
from pathlib import Path
from dataclasses import dataclass


m_mu = 1.883531627e-28 * u.kg


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

        Returns:
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

        Returns:
            (float or numpy.ndarray): Total mass energy density [g / cm^3]
        """
        e_eq = self.equil_energy_per_nucleon(n) * u.MeV  # e_eq [MeV]
        m_n_MeV = (cc.m_n * cc.c**2).to(u.MeV)  # n mass [MeV]
        rho = (n * u.fm**-3) * (e_eq + m_n_MeV) / cc.c**2
        return rho.to_value(u.g / u.cm**3)

    def pressure_equilibrium(self, n):
        """
        Analytical fit of the pressure in the equilibrium condiguration
        See eq. (C4) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n (float or numpy.ndarray): mean baryon number density [fm^-3]

        Returns:
            (float or numpy.ndarray): pressure in the eq. conf. [MeV fm^-3]
        """
        p_values = self.load_p_i_coefficients("TableC2")
        (
            p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
            p14, p15, p16, p17, p18, p19, p20, p21, p22, p23
        ) = p_values
        K = -33.2047  # for the pressure to be in [MeV fm^-3]
        rho = self.tot_mass_ener_dens(n)  # rho in [g / cm^3]
        xi = np.log10(rho)
        part1 = (p1 + p2 * xi + p3 * xi**3) / (1.0 + p4 * xi)
        part2 = 1.0 / (np.exp(p5 * (xi - p6)) + 1.0)
        part3 = p7 + p8 * xi
        part4 = 1.0 / (np.exp(p9 * (p6 - xi)) + 1.0)
        part5 = p10 + p11 * xi
        part6 = 1.0 / (np.exp(p12 * (p13 - xi)) + 1.0)
        part7 = p14 + p15 * xi
        part8 = 1.0 / (np.exp(p16 * (p17 - xi)) + 1.0)
        part9 = p18 / (1.0 + (p20 * (xi - p19))**2)
        part10 = p21 / (1.0 + (p23 * (xi - p22))**2)
        log10_P = (
                K + part1 * part2
                + part3 * part4
                + part5 * part6
                + part7 * part8
                + part9 + part10
                )
        P_MeVfm3 = 10.0**log10_P * u.MeV / u.fm**3
        return P_MeVfm3.value

    def e_fraction_core_Ye(self, n):
        """
        Parameterization of the electron fraction Y_e in the core
        See eq. (C17) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n (float or numpy.ndarray): baryon number density [fm^-3]

        Returns:
            (float or numpy.ndarray): e fraction Y_e in the core
        """
        p_values = self.load_p_i_coefficients("TableC7")
        p1, p2, p3, p4, p5, p6, p7 = p_values
        numerator = p1 + p2 * n + p6 * n**(3.0 / 2.0) + p3 * n**p7
        denominator = 1.0 + p4 * n**(3.0 / 2.0) + p5 * n**p7
        return numerator / denominator

    def relat_factor_fermi_surf(self, n_lept, m_lept):
        """
        Relativity factor at the Fermi surface of the e's or mu's
        See eq. (C15) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n_lept (float or numpy.ndarray): e or mu number density [fm^-3]
            m_lept (float): e or mu mass [kg]

        Returns:
            (float or numpy.ndarray): rel. factor at the Fermi surface
        """
        n_fm3 = n_lept * u.fm**-3
        m_kg = m_lept * u.kg
        part1 = (cc.hbar / (m_kg * cc.c)).to(u.fm)
        part2 = (3.0 * np.pi**2 * n_fm3)**(1.0 / 3.0)
        return (part1 * part2).value

    def muon_num_dens_core(self, n_e):
        """
        Parameterization of the muon number density in the core
        See eq. (C16) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n_e (numpy.ndarray): e number densities in core [fm^-3]

        Returns:
            (numpy.ndarray): mu number densities in core [fm^-3]
        """
        n_mu_list = []
        xe = self.relat_factor_fermi_surf(n_e, cc.m_e.value)
        part1 = 1.0 / (3.0 * np.pi**2)
        part2 = (cc.m_e * cc.c / cc.hbar).to(u.fm**-1).value
        for x in xe:
            if (1.0 + xe**2) > (((cc.m_mu / cc.m_e)**2).value):
                part3 = np.sqrt(1.0 + xe**2 - (m_mu.value / cc.m_e.value)**2) 
                n_mu_list.append(part1 * (part2 * part3)**3)
            else:
                n_mu_list.append(0.0)
        return np.array(n_mu_list)

    def fermi_momentum_kF(self, n):
        kF = (3.0 * np.pi**2 * n / 2.0)**(1.0 / 3.0)
        return kF

def load_eos(name: str) -> BSkEOS:
    """
    Loads a given BSk class

    Args:
        name (str): "BSk22", "BSk24", "BSk25" or "BSk26"

    Returns:
        (class): BSk22, BSk24, BSk25 or BSk26 class
    """
    return BSkEOS(name=name, params=bsk_params_raw[name])
