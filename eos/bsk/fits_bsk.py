import yaml
import numpy as np
from astropy import constants as cc
from astropy import units as u
from fractions import Fraction
from pathlib import Path
from dataclasses import dataclass


# Muon mass [kg]
m_mu = 1.883531627e-28 * u.kg


# Load .yaml file with BSk22, BSk24, BSk25, BSk26 params.
this_files_path = Path(__file__).resolve().parent
bsk_yaml_path = this_files_path / "parameters_bsk.yaml"
with open(bsk_yaml_path, "r") as f:
    bsk_params_raw = yaml.safe_load(f)


@dataclass(frozen=True)
class BSkEOS:
    name: str
    params: dict

    def load_p_i_coefficients(self, table_name):
        """
        Function to load coeffs. p_i in Tables C1-C10 in the BSk EOS
        eos.bsk.parameters_bsk.yaml file (TABLEII not included)

        Args:
            table_name (str): header in .yaml file ("TableC1", "TableC2", etc.)

        Returns:
            (generator): generator yielding p1, p2, ... in corresp. table
        """
        p = self.params[table_name]
        return (p[f"p{i}"] for i in range(1, len(p) + 1))

    # ---- analytical fits ----
    def equil_energy_per_nucleon(self, n):
        """
        Equilibrium energy per nucleon.
        See eqs. (C1) and (C2) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n (float or numpy.ndarray): mean baryon number density [fm^-3]

        Returns:
            (float or numpy.ndarray): Equilibrium energy per nucleon [MeV]
        """
        p_values = self.load_p_i_coefficients("TableC1")
        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14 = p_values

        e_gr = -9.1536  # MeV
        # factors appearing in e_eq
        w1 = 1.0 / (1.0 + p9 * n)  # See eq. (C2)
        w2 = 1.0 / (1.0 + (p13 * n)**p14)  # See eq. (C2)
        A = (p1 * n)**(7.0 / 6.0)
        B = 1.0 + np.sqrt(p2 * n)
        C = 1.0 + np.sqrt(p3 * n)
        D = 1.0 + np.sqrt(p4 * n)
        E = 1.0 + np.sqrt(p5 * n)
        F = p6 * n**p7
        G = 1.0 + p8 * n
        H = (1 - w1) * w2
        II = (p10 * n)**p11
        J = 1.0 + p12 * n
        K = 1.0 - w2
        return e_gr + A*D*w1 / (B*C*E) + F*G*H + II*K/J  # MeV

    def deriv_eq_e_per_nucleon(self, n):
        """
        Derivative of equilib. ener. per nucleon in eq. (C1) in
        https://doi.org/10.1093/mnras/sty2413 with resp. to mean bar. num. den.

        Args:
            n (float or numpy.ndarray): mean baryon number density [fm^-3]

        Returns:
            (float or numpy.ndarray): aforementioned derivative [MeV fm^3]
        """
        p_values = self.load_p_i_coefficients("TableC1")
        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14 = p_values

        # factors appearing in e_eq
        w1 = 1.0 / (1.0 + p9 * n)  # See eq. (C2)
        w2 = 1.0 / (1.0 + (p13 * n)**p14)  # See eq. (C2)
        A = (p1 * n)**(7.0 / 6.0)
        B = 1.0 + np.sqrt(p2 * n)
        C = 1.0 + np.sqrt(p3 * n)
        D = 1.0 + np.sqrt(p4 * n)
        E = 1.0 + np.sqrt(p5 * n)
        F = p6 * n**p7
        G = 1.0 + p8 * n
        H = (1 - w1) * w2
        II = (p10 * n)**p11
        J = 1.0 + p12 * n
        K = 1.0 - w2
        # derivatives with respect to n of the factors above
        dw1 = -p9 * w1**2
        dw2 = -p13 * p14 * (p13 * n)**(p14 - 1.0) * w2**2
        dA = (7.0 * A) / (6.0 * n)
        dB = p2 / (2.0 * np.sqrt(p2 * n))
        dC = p3 / (2.0 * np.sqrt(p3 * n))
        dD = p4 / (2.0 * np.sqrt(p4 * n))
        dE = p5 / (2.0 * np.sqrt(p5 * n))
        dF = p6 * p7 * n**(p7 - 1.0)
        dG = p8
        dH = dw2 - dw1*w2 - w1*dw2
        dII = p10 * p11 * (p10 * n)**(p11 - 1.0)
        dJ = p12
        dK = -dw2
        # sum these to get the derivative:
        term1 = (1.0 / (B*C*E)**2) * B*C*E * (dA*D*w1 + A*(dD*w1 + D*dw1))
        term2 = -(1.0 / (B*C*E)**2) * A*D*w1 * (dB*C*E + B*(dC*E + C*dE))
        term3 = dF*G*H + F*dG*H + F*G*dH
        term4 = (1.0 / J**2) * (J*dII*K + J*II*dK - II*K*dJ)
        return term1 + term2 + term3 + term4

    def tot_mass_ener_dens(self, n):
        """
        Total mass-energy density
        See eq. (4) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n (float or numpy.ndarray): mean baryon number density [fm^-3]

        Returns:
            (float or numpy.ndarray): Total mass energy density [g / cm^3]
        """
        e_eq = self.equil_energy_per_nucleon(n) * u.MeV  # e_eq [MeV]
        n_fm3 = n * u.fm**-3
        return (n_fm3 * (e_eq / cc.c**2 + cc.m_n)).to_value(u.g / u.cm**3)

    def pressure_equilibrium(self, n):
        """
        Analytical fit of the pressure in the equilibrium configuration
        See eq. (C4) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n (float or numpy.ndarray): mean baryon number density [fm^-3]

        Returns:
            (float or numpy.ndarray): pressure in the eq. conf. [MeV fm^-3]
        """
        def exp_subfunc(xi, pi, pd):
            return 1.0 / (np.exp(pi * (pd - xi)) + 1.0)

        p_values = self.load_p_i_coefficients("TableC2")
        (
            p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
            p14, p15, p16, p17, p18, p19, p20, p21, p22, p23
        ) = p_values

        rho = self.tot_mass_ener_dens(n)  # rho(n) in [g / cm^3]
        xi = np.log10(rho)
        # Factors appearing in eq. (C4)
        K = -33.2047  # for the pressure to be in [MeV fm^-3]
        A = p1 + p2*xi + p3*xi**3
        B = 1.0 + p4*xi
        C = 1.0 / (np.exp(p5 * (xi - p6)) + 1.0)
        C1 = exp_subfunc(xi, p9, p6)
        C2 = exp_subfunc(xi, p12, p13)
        C3 = exp_subfunc(xi, p16, p17)
        D1 = p7 + p8*xi
        D2 = p10 + p11*xi
        D3 = p14 + p15*xi
        E1 = p18 / (1.0 + (p20 * (xi - p19))**2)
        E2 = p21 / (1.0 + (p23 * (xi - p22))**2)
        log10_P = (
                K + A*C/B  # line 1
                + D1 * C1  # line 2
                + D2 * C2  # line 3
                + D3 * C3  # line 4
                + E1 + E2  # line 5
                )
        P_MeVfm3 = 10.0**log10_P * u.MeV / u.fm**3
        return P_MeVfm3.value

    def deriv_pressure_equilibrium(self, n):
        """
        Total derivative of the pressure in the equil. config. with respect
        to the mean baryon number density. See eqs. (C4), (C1) and (4) in
        https://doi.org/10.1093/mnras/sty2413

        Args:
            n (float or numpy.ndarray): mean baryon number density [fm^-3]

        Returns:
            (float or numpy.ndarray): deriv. of pressure in equil. conf. [MeV]
        """
        def exp_subfunc(xi, pi, pd):
            return 1.0 / (np.exp(pi * (pd - xi)) + 1.0)

        def d_exp_subfuc(xi, pi, pd):
            return exp_subfunc(xi, pi, pd)**2 * pi * np.exp(pi * (pd - xi))

        p_values = self.load_p_i_coefficients("TableC2")
        (
            p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
            p14, p15, p16, p17, p18, p19, p20, p21, p22, p23
        ) = p_values

        rho = self.tot_mass_ener_dens(n)  # rho(n) in [g / cm^3]
        xi = np.log10(rho)

        # dlog_10(P) / dxi
        A = p1 + p2*xi + p3*xi**3
        B = 1.0 + p4*xi
        C = 1.0 / (np.exp(p5 * (xi - p6)) + 1.0)
        C1 = exp_subfunc(xi, p9, p6)
        C2 = exp_subfunc(xi, p12, p13)
        C3 = exp_subfunc(xi, p16, p17)
        D1 = p7 + p8*xi
        D2 = p10 + p11*xi
        D3 = p14 + p15*xi
        E1 = p18 / (1.0 + (p20 * (xi - p19))**2)
        E2 = p21 / (1.0 + (p23 * (xi - p22))**2)

        dA = p2 + 3.0 * p3 * xi**2
        dB = p4
        dC = -C**2 * p5 * np.exp(p5 * (xi - p6))
        dC1 = d_exp_subfuc(xi, p9, p6)
        dC2 = d_exp_subfuc(xi, p12, p13)
        dC3 = d_exp_subfuc(xi, p16, p17)
        dD1 = p8
        dD2 = p11
        dD3 = p15
        dE1 = -E1**2 * 2.0 * p20 * (p20 * (xi - p19)) / p18
        dE2 = -E2**2 * 2.0 * p23 * (p23 * (xi - p22)) / p21
        dlog10_P_dxi = (
                  ((dA*C + A*dC)*B - A*C*dB) / B**2  # line 1
                  + dD1*C1 + D1*dC1  # line 2
                  + dD2*C2 + D2*dC2  # line 3
                  + dD3*C3 + D3*dC3  # line 4
                  + dE1 + dE2  # line 5
                  )  # dlog_10(P) / dxi

        P_MeVfm3 = self.pressure_equilibrium(n) * u.MeV / u.fm**3  # P
        dP_dxi = np.log(10.0) * P_MeVfm3 * dlog10_P_dxi  # dP/dxi [MeV fm^-3]
        dxi_drho = (1.0 / (rho * np.log(10.0))) * u.cm**3 / u.g  # dxi/drho
        deeq_dn = self.deriv_eq_e_per_nucleon(n) * u.MeV * u.fm**3
        drho_dn = (
           self.equil_energy_per_nucleon(n) * u.MeV / cc.c**2
           + cc.m_n
           + n * u.fm**-3 * deeq_dn / cc.c**2
           )  # drho/dn (see eq. (4)) [g]

        return (dP_dxi * dxi_drho * drho_dn).to_value(u.MeV)

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
        return (numerator / denominator) * n  # n_e = Y_e * n

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
        # if n float, returns arr w/ that float, otherw same arr
        n_arr = np.asarray(n)
        n_e_arr = self.e_num_dens_core(n_arr)  # fm^-3
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
        return 1.0 - 2.0 * n_p / n  # Y_p = n_p / n

    def F_x(self, eta, x):
        """
        F_x(eta) in eq. (33) in https://doi.org/10.1093/mnras/sty2413
        used in core calculations

        Args:
            eta (float or numpy.ndarray): isospin asymmetry param. eta [adim]
            x (float): either 5.0 / 3.0 or 8.0 / 3.0
        Returns:
            (float or numpy.ndarray): F_x(eta) in eq. (33) [adim]
        """
        return 0.5 * ((1.0 + eta)**x + (1.0 - eta)**x)

    def skyrme_pressure_core(self, n):
        """
        Skyrme pressure in the core
        See eq. (35) in https://doi.org/10.1093/mnras/sty2413
        and parameters in https://doi.org/10.1103/PhysRevC.88.024308

        Args:
            n (float or numpy.ndarray): baryon number density in core [fm^-3]
        Returns:
            (float or numpy.ndarray): P_Skyrme in core [MeV fm^-3]
            (MIGHT RETURN nan VALUES OUTSIDE THE CORE (AT LOWER DENSITIES))
        """
        n_fm3 = n * u.fm**-3
        eta = self.isospin_asym_eta(n)  # [adim]
        kF = self.fermi_momentum_kF(n) * u.fm**-1
        F53 = self.F_x(eta, 5.0 / 3.0)  # [adim]
        F83 = self.F_x(eta, 8.0 / 3.0)  # [adim]

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

        part1 = (cc.hbar * kF)**2 * n_fm3 / 10.0
        part2 = (1.0 + eta)**(5.0 / 3.0) / cc.m_n
        part3 = (1.0 - eta)**(5.0 / 3.0) / cc.m_p

        part4 = t0 * n_fm3**2 / 8.0
        part5 = 3.0 - (1.0 + 2.0 * x0) * eta**2

        part6 = t1 * (n_fm3 * kF)**2 / 8.0
        part7 = (2.0 + x1) * F53
        part8 = (0.5 + x1) * F83

        part9 = (n_fm3 * kF)**2 / 8.0
        part10 = (2.0*t2 + t2x2) * F53
        part11 = (0.5*t2 + t2x2) * F83

        part12 = (al + 1.0) / 48.0 * t3 * n_fm3**(al + 2.0)
        part13 = 3.0 - (1.0 + 2.0 * x3) * eta**2

        part14 = (3.0 * be + 5.0) / 40.0 * t4 * n_fm3**(be + 2.0) * kF**2
        part15 = (2.0 + x4) * F53
        part16 = (0.5 + x4) * F83

        part17 = (3.0 * ga + 5.0) / 40.0 * t5 * n_fm3**(ga + 2.0) * kF**2
        part18 = (2.0 + x5) * F53
        part19 = (0.5 + x5) * F83

        return ((
            part1 * (part2 + part3)
            + part4 * part5
            + part6 * (part7 - part8)
            + part9 * (part10 + part11)
            + part12 * part13
            + part14 * (part15 - part16)
            + part17 * (part18 + part19)
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
        return (cc.hbar / (m_lept * u.kg * cc.c)).to_value(u.fm)

    def x_function(self, n_lept, m_lept):
        """
        Function x in eq. (B1) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n_lept (float or numpy.ndarray): (e or mu) num. dens. [fm^-3]
            m_lept (float): lepton (e or mu) mass [kg]

        Returns:
            (float or numpy.ndarray): x in eq. (B1) [adim]
        """
        lam = self.compton_wlength(m_lept)  # fm
        return lam * (3.0 * np.pi**2 * n_lept)**(1.0 / 3.0)

    def f_function(self, n_lept, m_lept):
        """
        Function f in eq. (B14) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n_lept (float or numpy.ndarray): (e or mu) num. dens. [fm^-3]
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
            n_lept (float or numpy.ndarray): (e or mu) num. dens. [fm^-3]
            m_lept (float): lepton (e or mu) mass [kg]

        Returns:
            (float or numpy.ndarray): P^kin in eq. (B15) [MeV fm^-3]
        """
        f = self.f_function(n_lept, m_lept)  # [adim]
        m_kg = m_lept * u.kg
        lam = self.compton_wlength(m_lept) * u.fm
        numerator = m_kg * cc.c**2 * f
        denominator = 24.0 * np.pi**2 * lam**3
        return (numerator / denominator).to_value(u.MeV * u.fm**-3)

    def auxiliar_function(self, x):
        """
        Auxiliar function of x: sqrt(1+x**2)/x - arcsinh(x)/x**2

        Args:
            (float or numpy.ndarray): e (or mu) x as in eq. (B1)
            in https://doi.org/10.1093/mnras/sty2413

        Returns:
            (float or numpy.ndarray): aforementioned function of x

        """
        x = np.asarray(x)
        val = np.sqrt(1.0 + x**2) / x - np.arcsinh(x) / x**2
        val = np.where(x == 0.0, 0.0, val)
        return val.item() if np.isscalar(x) else val

    def auxiliar_derivative(self, x):
        """
        Derivative with respect to x of sqrt(1+x**2)/x - arcsinh(x)/x**2

        Args:
            (float or numpy.ndarray): e (or mu) x as in eq. (B1)
            in https://doi.org/10.1093/mnras/sty2413 [adim]

        Returns:
            (float or numpy.ndarray): aforementioned derivative

        """
        x = np.asarray(x)
        val = (2.0 / x**2) * (-1.0 / np.sqrt(1.0 + x**2) + np.arcsinh(x) / x)
        val = np.where(x == 0.0, 0.0, val)
        return val.item() if np.isscalar(x) else val

    def exchange_ener_dens(self, n_lept, m_lept):
        """
        Exchange energy density (for e or mu) in eq. (B9)
        in https://doi.org/10.1093/mnras/sty2413

        Args:
            n_lept (float or numpy.ndarray): (e or mu) num. dens. [fm^-3]
            m_lept (float): lepton (e or mu) mass [kg]

        Returns:
            (float or numpy.ndarray): E^ex in eq. (B9) [MeV fm^-3]
        """
        m_kg = m_lept * u.kg
        lam = self.compton_wlength(m_lept) * u.fm
        x = self.x_function(n_lept, m_lept)  # [adim]

        part1 = cc.alpha.value * m_kg * cc.c**2 * x**4
        part2 = 4.0 * (np.pi * lam)**3
        part3 = 1.0 - (3.0 / 2.0) * self.auxiliar_function(x)**2
        result = -(part1 / part2) * part3
        # Impose result = 0.0 when x=0 (when n_lept=0):
        # result = np.where(x == 0, 0.0, result)
        return result.to_value(u.MeV * u.fm**-3)

    def lepton_exchange_pressure(self, n_lept, m_lept):
        """
        Exchange lepton (e or mu) pressure P^ex in eq. (B17)
        in https://doi.org/10.1093/mnras/sty2413

        Args:
            n_lept (float or numpy.ndarray): (e or mu) num. dens. [fm^-3]
            m_lept (float): lepton (e or mu) mass [kg]

        Returns:
            (float or numpy.ndarray): P^ex in eq. (B17) [MeV fm^-3]
        """
        m_kg = m_lept * u.kg
        e_ex = self.exchange_ener_dens(n_lept, m_lept) * u.MeV * u.fm**-3
        lam = self.compton_wlength(m_lept) * u.fm
        x = self.x_function(n_lept, m_lept)  # [adim]

        part1 = e_ex / 3.0
        part2 = cc.alpha.value * m_kg * cc.c**2 * x**3
        part3 = 2.0 * (np.pi * lam)**3
        part4 = 1.0 / np.sqrt(1.0 + x**2) - np.arcsinh(x) / x
        part5 = self.auxiliar_function(x)

        fac = u.MeV * u.fm**-3
        result = part1 - part2 * part4 * part5 / part3
        result = np.where(x == 0.0, 0.0, result)
        scalar_res = (result.item()).to_value(fac)
        return scalar_res if np.isscalar(x) else result.to_value(fac)

    def deriv_lept_kin_p_const_comp(self, n, n_lept, m_lept):
        """
        Partial deriv. of e (or mu) kinet. pressure at const. e (or mu)
        composition with respect to the baryon number density.
        Deriv. taken from eq. (B15) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n (float or numpy.ndarray): baryon number density [fm^-3]
            n_lept (float or numpy.ndarray): (e or mu) num. dens. [fm^-3]
            m_lept (float): lepton (e or mu) mass [kg]

        Returns:
            (float or numpy.ndarray): aforementioned deriv. [MeV]
        """
        n_fm3 = n * u.fm**-3
        x = self.x_function(n_lept, m_lept)  # [adim]
        m_kg = m_lept * u.kg
        lam = self.compton_wlength(m_lept) * u.fm
        part1 = m_kg * cc.c**2 / (24.0 * np.pi**2 * lam**3)
        part2 = 8.0 * x**5 / (3.0 * n_fm3 * np.sqrt(1.0 + x**2))
        return (part1 * part2).to_value(u.MeV)

    def deriv_exch_e_dens_const_comp(self, n, n_lept, m_lept):
        """
        Partial deriv. of e (or mu) exchange ener. dens. at const. e (or mu)
        composition with respect to the baryon number density.
        Deriv. taken from eq. (B9) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n (float or numpy.ndarray): baryon number density [fm^-3]
            n_lept (float or numpy.ndarray): (e or mu) num. dens. [fm^-3]
            m_lept (float): lepton (e or mu) mass [kg]

        Returns:
            (float or numpy.ndarray): aforementioned deriv. [MeV]
        """
        n_fm3 = n * u.fm**-3
        m_kg = m_lept * u.kg
        lam = self.compton_wlength(m_lept) * u.fm
        x = self.x_function(n_lept, m_lept)  # [adim]
        h = self.auxiliar_function(x)  # [adim]
        dhdx = self.auxiliar_derivative(x)  # [adim]

        part1 = - cc.alpha * m_kg * cc.c**2 / (4.0 * np.pi**3 * lam**3)
        part2 = x**4 / n_fm3
        part3 = 4.0 / 3.0 - 2.0 * h**2 - x * h * dhdx
        return (part1 * part2 * part3).to_value(u.MeV)

    def deriv_lept_exch_p_const_comp(self, n, n_lept, m_lept):
        """
        Partial deriv. of e (or mu) lept. exch. p at const. e (or mu)
        composition with respect to the baryon number density.
        Deriv. taken from eq. (B17) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n (float or numpy.ndarray): baryon number density [fm^-3]
            n_lept (float or numpy.ndarray): (e or mu) num. dens. [fm^-3]
            m_lept (float): lepton (e or mu) mass [kg]

        Returns:
            (float or numpy.ndarray): aforementioned deriv. [MeV]
        """
        lam = self.compton_wlength(m_lept) * u.fm
        m_kg = m_lept * u.kg
        deex_dn = self.deriv_exch_e_dens_const_comp(n, n_lept, m_lept) * u.MeV
        x = self.x_function(n_lept, m_lept)  # [adim]
        h = self.auxiliar_function(x)  # [adim]
        dhdx = self.auxiliar_derivative(x)  # [adim]
        dxdn = x / (3.0 * n * u.fm**-3)

        part1 = deex_dn / 3.0
        part2 = cc.alpha * m_kg * cc.c**2 / (2.0 * np.pi**3 * lam**3)
        part3 = x**2 * (2.0 - x**2 / (1.0 + x**2)) / np.sqrt(1.0 + x**2)
        part4 = -2.0 * x * np.arcsinh(x)
        part5 = x**3 / np.sqrt(1.0 + x**2) - x**2 * np.arcsinh(x)
        return ((
            part1 - dxdn * part2
            * (h * (part3 + part4) - dhdx * part5)).to_value(u.MeV)
            )

    def deriv_skyrme_p_const_comp(self, n):
        """
        Partial deriv. Skyrme pressure at constant composition
        with respect to the baryon number density (core calc.)
        Deriv. taken from eq. (35) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n (float or numpy.ndarray): baryon number density [fm^-3]

        Returns:
            (float or numpy.ndarray): aforementioned deriv. [MeV]
        """
        n_fm3 = n * u.fm**-3
        eta = self.isospin_asym_eta(n)  # [adim]
        kF = self.fermi_momentum_kF(n) * u.fm**-1
        F53 = self.F_x(eta, 5.0 / 3.0)  # [adim]
        F83 = self.F_x(eta, 8.0 / 3.0)  # [adim]

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

        part1 = (cc.hbar**2 / 10.0) * kF**2 * 5.0 / 3.0
        part2 = (1.0 + eta)**(5.0 / 3.0) / cc.m_n
        part3 = (1.0 - eta)**(5.0 / 3.0) / cc.m_p

        part4 = t0 * n_fm3 / 4.0
        part5 = 3.0 - (1.0 + 2.0 * x0) * eta**2

        part6 = t1 * n_fm3 * kF**2 / 3.0
        part7 = (2.0 + x1) * F53
        part8 = (0.5 + x1) * F83

        part9 = n_fm3 * kF**2 / 3.0
        part10 = (2.0*t2 + t2x2) * F53
        part11 = (0.5*t2 + t2x2) * F83

        part12 = (al + 1.0) / 48.0 * t3 * (al + 2.0) * n_fm3**(al + 1.0)
        part13 = 3.0 - (1.0 + 2.0 * x3) * eta**2

        part14 = (3.0 * be + 5.0) / 40.0 * t4 *\
            n_fm3**(be + 1.0) * kF**2 * (be + 8.0 / 3.0)
        part15 = (2.0 + x4) * F53
        part16 = (0.5 + x4) * F83

        part17 = (3.0 * ga + 5.0) / 40.0 * t5 *\
            n_fm3**(ga + 1.0) * kF**2 * (ga + 8.0 / 3.0)
        part18 = (2.0 + x5) * F53
        part19 = (0.5 + x5) * F83

        return ((
            part1 * (part2 + part3)
            + part4 * part5
            + part6 * (part7 - part8)
            + part9 * (part10 + part11)
            + part12 * part13
            + part14 * (part15 - part16)
            + part17 * (part18 + part19)
            ).to_value(u.MeV))

    def total_lept_pressure_core(self, n):
        """
        Total lepton pressure in the core.
        See eq. (36) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n (float or numpy.ndarray): baryon number density in core [fm^-3]
        Returns:
            (float or numpy.ndarray): total lept. P in core [MeV fm^-3]
            (MIGHT RETURN nan VALUES OUTSIDE THE CORE (AT LOWER DENSITIES))
        """
        # electron contribution
        n_e = self.e_num_dens_core(n)
        p_kin_e = self.lepton_kin_pressure(n_e, cc.m_e.value)
        p_ex_e = self.lepton_exchange_pressure(n_e, cc.m_e.value)
        # muon contribution
        n_mu = self.mu_num_dens_core(n)
        p_kin_mu = self.lepton_kin_pressure(n_mu, m_mu.value)
        p_ex_mu = self.lepton_exchange_pressure(n_mu, m_mu.value)
        return p_kin_e + p_kin_mu + p_ex_e + p_ex_mu

    def total_pressure_core(self, n):
        """
        Total pressure in the core.
        See eq. (34) in https://doi.org/10.1093/mnras/sty2413

        Args:
            n (float or numpy.ndarray): baryon number density in core [fm^-3]
        Returns:
            (float or numpy.ndarray): total P in core [MeV fm^-3]
            (MIGHT RETURN nan VALUES OUTSIDE THE CORE (AT LOWER DENSITIES))
        """
        p_lept_total_core = self.total_lept_pressure_core(n)
        p_skyrme_core = self.skyrme_pressure_core(n)
        return p_lept_total_core + p_skyrme_core

    def total_deriv_pressure_core(self, n):
        """
        Partial derivative of the total pressure in the core at constant
        composition and with respect to the baryon density

        Args:
            n (float or numpy.ndarray): baryon number density in core [fm^-3]
        Returns:
            (float or numpy.ndarray): aforementioned partial deriv. [MeV]
            (MIGHT RETURN nan VALUES OUTSIDE THE CORE (AT LOWER DENSITIES))
        """
        n_e = self.e_num_dens_core(n)
        n_mu = self.mu_num_dens_core(n)

        dpskyrme_dn = self.deriv_skyrme_p_const_comp(n)
        dpexe_dn = self.deriv_lept_exch_p_const_comp(n, n_e, cc.m_e.value)
        dpexmu_dn = self.deriv_lept_exch_p_const_comp(n, n_mu, m_mu.value)
        dpkine_dn = self.deriv_lept_kin_p_const_comp(n, n_e, cc.m_e.value)
        dpkinmu_dn = self.deriv_lept_kin_p_const_comp(n, n_mu, m_mu.value)
        return dpskyrme_dn + dpexe_dn + dpexmu_dn + dpkine_dn + dpkinmu_dn


def load_eos(name: str) -> BSkEOS:
    """
    Loads a given BSk class

    Args:
        name (str): "BSk22", "BSk24", "BSk25" or "BSk26"

    Returns:
        (class): BSk22, BSk24, BSk25 or BSk26 class
    """
    return BSkEOS(name=name, params=bsk_params_raw[name])
