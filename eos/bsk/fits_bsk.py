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
with open(bsk_yaml_path, "r") as f:
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

    def e_num_dens_core(self, n):
        """
        Electron num. density in core via the parameterization of Y_e
        in eq. (C17) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n (float or numpy.ndarray): baryon number density [fm^-3]

        Returns:
            (float or numpy.ndarray): e num. density in the core [fm^-3]
        """
        p_values = self.load_p_i_coefficients("TableC7")
        p1, p2, p3, p4, p5, p6, p7 = p_values
        numerator = p1 + p2 * n + p6 * n**(3.0 / 2.0) + p3 * n**p7
        denominator = 1.0 + p4 * n**(3.0 / 2.0) + p5 * n**p7
        return n * (numerator / denominator)  # n_e = Y_e * n

    def relat_factor_fermi_surf(self, n_lept, m_lept):
        """
        Relativity factor at the Fermi surface of the e's or mu's
        See eq. (C15) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n_lept (float or numpy.ndarray): e or mu number density [fm^-3]
            m_lept (float): e or mu mass [kg]

        Returns:
            (float or numpy.ndarray): rel. factor at the Fermi surface [adim]
        """
        n_fm3 = n_lept * u.fm**-3
        m_kg = m_lept * u.kg
        part1 = (cc.hbar / (m_kg * cc.c)).to(u.fm)
        part2 = (3.0 * np.pi**2 * n_fm3)**(1.0 / 3.0)
        return (part1 * part2).value

    def mu_num_dens_core(self, n):
        """
        Parameterization of the muon number density in the core
        See eq. (C16) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n (float or numpy.ndarray): baryon num. densities in core [fm^-3]

        Returns:
            (float or numpy.ndarray): mu number densities in core [fm^-3]
        """
        n_arr = np.asarray(n)
        # if n_e is a float, returns arr with that float, otherw same arr
        n_e_arr = self.e_num_dens_core(n_arr)
        xe = self.relat_factor_fermi_surf(n_e_arr, cc.m_e.value)
        part1 = 1.0 / (3.0 * np.pi**2)
        part2 = (cc.m_e * cc.c / cc.hbar).to_value(u.fm**-1)
        # n_mu != 0 only if 1 + xe**2 > (m_mu/m_e)**2
        m_ratio_2 = ((m_mu / cc.m_e).value)**2
        mask = 1.0 + xe**2 > m_ratio_2
        part3 = np.zeros_like(xe)
        part3[mask] = np.sqrt(1.0 + xe[mask]**2 - m_ratio_2)

        n_mu = part1 * (part2 * part3)**3
        return n_mu.item() if np.isscalar(n_mu) else n_mu

    def fermi_momentum_kF(self, n):
        """
        Fermi momentum k_F (used in core calculations)
        See eq. (31) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n (float or numpy.ndarray): baryon number density in core [fm^-3]

        Returns:
            (float or numpy.ndarray): Fermi momentum k_F [fm^-1]
        """
        return (3.0 * np.pi**2 * n / 2.0)**(1.0 / 3.0)

    def p_num_dens_core(self, n):
        """
        Proton num. density in core assuming Y_p = Y_e + Y_mu
        in eq. (28) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n (float or numpy.ndarray): baryon number density [fm^-3]

        Returns:
            (float or numpy.ndarray): proton density in the core [fm^-3]
        """
        n_e = self.e_num_dens_core(n)
        n_mu = self.mu_num_dens_core(n)
        return n_e + n_mu

    def isospin_asym_eta(self, n):
        """
        Isospin asymmetry parameter eta = (n_n - n_p) / n (used in core calcs.)
        See eq. (32) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n (float or numpy.ndarray): baryon number density in core [fm^-3]

        Returns:
            (float or numpy.ndarray): isospin asymmetry param. eta [adim]
        """
        n_p = self.p_num_dens_core(n)
        return 1.0 + 2.0 * n_p / n

    def F_x(self, n, x):
        """
        F_x(eta) in eq. (33) in https://doi.org/10.1093/mnras/sty2413
        used in core calculations

        Args:
            n (float or numpy.ndarray): baryon number density in core [fm^-3]
        Returns:
            (float or numpy.ndarray): F_x(eta) in eq. (33) [adim]
        """
        eta = self.isospin_asym_eta(n)
        return 0.5 * ((1.0 + eta)**x + (1.0 - eta)**x)

    def skyrme_pressure_core(self, n):
        """
        Skyrme pressure in the core
        See eq. (35) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n (float or numpy.ndarray): baryon number density in core [fm^-3]
        Returns:
            (float or numpy.ndarray): P_Skyrme in core [MeV fm^-3]
        """
        eta = self.isospin_asym_eta(n)
        kF = self.fermi_momentum_kF(n) * u.fm**-1
        F53 = self.F_x(n, 5.0 / 3.0)
        F83 = self.F_x(n, 8.0 / 3.0)
        # DELETE AFTER DEBUGGINGGGGGGGGGGGGGGG 
        print("eta", eta)
        print("kF", kF)
        print("F53", F53)
        print("F83", F83)

        p = self.params["TABLEII"]
        al = float(Fraction(p["alpha"]))
        be = float(Fraction(p["beta"]))
        ga = float(Fraction(p["gamma"]))
        t0 = p["t0"] * u.MeV * u.fm**3
        t1 = p["t1"] * u.MeV * u.fm**5
        t2 = p["t2"] * u.MeV * u.fm**5
        t3 = p["t3"] * u.MeV * u.fm**(3.0 + 3.0*al)
        t4 = p["t4"] * u.MeV * u.fm**(5.0 + 3.0*be)
        t5 = p["t5"] * u.MeV * u.fm**(5.0 + 3.0*ga)
        t2x2 = p["t2x2"] * u.MeV * u.fm**5
        x0 = p["x0"]
        x1 = p["x1"]
        x3 = p["x3"]
        x4 = p["x4"]
        x5 = p["x5"]
        # DELETE AFTER DEBUGGINGGGGGGGGGGGGGGG 
        print("Params:", al, be, ga, t0, t1, t2, t3, t4, t5, t2x2, x0, x1, x3, x4, x5)

        n_fm3 = n * u.fm**-3
        part1 = (cc.hbar * kF)**2 * n_fm3 / 10.0
        part2 = (1.0 + eta)**(5.0 / 3.0) / cc.m_n
        part3 = (1.0 - eta)**(5.0 / 3.0) / cc.m_p

        part4 = t0 * n_fm3**2 / 8.0
        part5 = 3.0 - (1.0 + 2.0 * x0) * eta**2

        part6 = t1 * (n_fm3 * kF)**2 / 8.0
        part7 = (2.0 + x1) * F53
        part8 = (0.5 + x1) * F83

        part9 = (n_fm3 * kF)**2 / 8.0
        part10 = (2.0 * t2 + t2x2) * F53
        part11 = (0.5 * t2 + t2x2) * F83

        part12 = (al + 1.0) / 48.0 * t3 * n_fm3**(al + 2.0)
        part13 = 3.0 - (1.0 + 2.0 * x3) * eta**2

        part14 = (3.0 * be + 5.0) / 40.0 * t4 * n_fm3**(be + 2.0) * kF**2
        part15 = (2.0 + x4) * F53
        part16 = (0.5 + x4) * F83

        part17 = (3.0 * ga + 5.0) / 40.0 * t5 * n_fm3**(ga + 2.0) * kF**2
        part18 = (2.0 + x5) * F53
        part19 = (0.5 + x5) * F83
        # DELETE AFTER DEBUGGINGGGGGGGGGGGGGGG 
        print(part1)
        print(part2)
        print(part3)
        print(part4)
        print(part5)
        print(part6)
        print(part7)
        print(part8)
        print(part9)
        print(part10)
        print(part11)
        print(part12)
        print(part13)
        print(part14)
        print(part15)
        print(part16)
        print(part17)
        print(part18)
        print(part19)
        return ((
            part1 * (part2 + part3)
            + part4 * part5
            + part6 * (part7 - part8)
            + part9 * (part10 + part11)
            + part12 * part13
            + part14 * (part15 - part16)
            + part17 * (part18 - part19)
            ).to_value(u.MeV / u.fm**3))

    def compton_wlength(self, m_lept):
        """
        Compton wavelength of the lepton (e or mu)
        See below eq. (B1) in https://doi.org/10.1093/mnras/sty2413

        Args:
            m_lept (float): lepton (e or mu) mass [kg]

        Returns:
            (float): associated Compton wavelength [fm]
        """
        m_kg = m_lept * u.kg
        return (cc.hbar / (m_kg * cc.c)).to_value(u.fm)

    def x_function(self, n_lept, m_lept):
        """
        Function x in eq. (B1) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n_lept (float or numpy.ndarray): (e or mu) num. dens. [fm^-e]
            m_lept (float): lepton (e or mu) mass [kg]

        Returns:
            (float or numpy.ndarray): x in eq. (B1) [adim]
        """
        lam = self.compton_wlength(m_lept)
        return lam * (3.0 * np.pi**2 * n_lept)**(1.0 / 3.0)

    def f_function(self, n_lept, m_lept):
        """
        Function f in eq. (B14) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n_lept (float or numpy.ndarray): (e or mu) num. dens. [fm^-e]
            m_lept (float): lepton (e or mu) mass [kg]

        Returns:
            (float or numpy.ndarray): f in eq. (B14) [adim]
        """

        x = self.x_function(n_lept, m_lept)
        return (
            (2.0 * x**3 - 3.0 * x) * np.sqrt(1.0 + x**2)
            + 3.0 * np.arcsinh(x)
            )

    def lepton_kin_pressure(self, n_lept, m_lept):
        """
        Kinetic lepton (e or mu) pressure P^kin in eq. (B15)
        in https://doi.org/10.1093/mnras/sty2413

        Args:
            n_lept (float or numpy.ndarray): (e or mu) num. dens. [fm^-e]
            m_lept (float): lepton (e or mu) mass [kg]

        Returns:
            (float or numpy.ndarray): P^kin in eq. (B15) [MeV fm^-3]
        """
        f = self.f_function(n_lept, m_lept)
        m_kg = m_lept * u.kg
        lam = self.compton_wlength(m_lept) * u.fm
        numerator = m_kg * cc.c**2 * f
        denominator = 24.0 * np.pi**2 * lam**3
        return (numerator / denominator).to_value(u.MeV * u.fm**-3)

    def exchange_ener_dens(self, n_lept, m_lept):
        """
        Exchange energy density (for e or mu) in eq. (B9)
        in https://doi.org/10.1093/mnras/sty2413

        Args:
            n_lept (float or numpy.ndarray): (e or mu) num. dens. [fm^-e]
            m_lept (float): lepton (e or mu) mass [kg]

        Returns:
            (float or numpy.ndarray): E^ex in eq. (B9) [MeV fm^-3]
        """
        m_kg = m_lept * u.kg
        lam = self.compton_wlength(m_lept) * u.fm
        x = self.x_function(n_lept, m_lept)

        part1 = cc.alpha.value * m_kg * cc.c**2 * x**4
        part2 = 4.0 * (np.pi * lam)**3
        part3 = np.sqrt(1.0 + x**2) / x - np.arcsinh(x) / x**2
        part4 = 1.0 - (3.0 / 2.0) * part3**2
        return (-part1 / part2 * part4).to_value(u.MeV * u.fm**-3)

    def lepton_exchange_pressure(self, n_lept, m_lept):
        """
        Exchange lepton (e or mu) pressure P^ex in eq. (B17)
        in https://doi.org/10.1093/mnras/sty2413

        Args:
            n_lept (float or numpy.ndarray): (e or mu) num. dens. [fm^-e]
            m_lept (float): lepton (e or mu) mass [kg]

        Returns:
            (float or numpy.ndarray): P^ex in eq. (B17) [MeV fm^-3]
        """
        m_kg = m_lept * u.kg
        e_ex = self.exchange_ener_dens(n_lept, m_lept) * u.MeV * u.fm**-3
        lam = self.compton_wlength(m_lept) * u.fm
        x = self.x_function(n_lept, m_lept)

        part1 = e_ex / 3.0
        part2 = cc.alpha.value * m_kg * cc.c**2 * x**3
        part3 = 2.0 * (np.pi * lam)**3
        part4 = 1.0 / np.sqrt(1.0 + x**2) - np.arcsinh(x) / x
        part5 = np.sqrt(1.0 + x**2) / x - np.arcsinh(x) / x**2
        return (
            (part1 - part2 * part4 * part5 / part3).to_value(u.MeV * u.fm**-3)
            )


def load_eos(name: str) -> BSkEOS:
    """
    Loads a given BSk class

    Args:
        name (str): "BSk22", "BSk24", "BSk25" or "BSk26"

    Returns:
        (class): BSk22, BSk24, BSk25 or BSk26 class
    """
    return BSkEOS(name=name, params=bsk_params_raw[name])
