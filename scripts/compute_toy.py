import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from pathlib import Path


from modes.spectral_methods.spectral import SpectralGrid
from modes.spectral_methods.ns_models_2x2_system import ToyModelsTseneklidou
import modes.spectral_methods.spurious as sp


# Filtering parametes
rel_diff_tol = 1e-2
mode_count_threshold = 60
mean_sq_threshold = 1e-1


class ComputeToyModes:
    def __init__(self, Gamma1, Cs2, M, R, alphaR,
                 points_per_domain, domains, r_shock=1.):
        self.Gamma1 = Gamma1
        self.Cs2 = Cs2
        self.M = M
        self.R = R
        self.alphaR = alphaR
        self.grid = SpectralGrid(r_shock, points_per_domain, domains)

    def solve_for_N02(self, N02):
        toy_params = {
            "Cs2": self.Cs2,
            "N2_0": N02,
            "Gamma1": self.Gamma1,
            "M_kg": self.M,
            "R_m": self.R,
            "alpha_R": self.alphaR
        }

        eq = ToyModelsTseneklidou(toy_params, self.grid)
        A, B = eq()
        
        # Solve eigenvalue problem
        lam, vecs = eig(A, B)

        # Sort according to |sigma| and above a sigma_min threshold
        sigma = np.sqrt(lam)
        sigma, vecs = sp.clean_and_sort(sigma, vecs, sigma_min=1e-8)

        return {
            "sigma": sigma,
            "vecs": vecs,
            "r": self.grid.r[:, 0],
            "N": self.grid.N,
            "alpha_R": eq.alpha_R
        }


def filtered_modes_for_N02(N02, solver_hi, solver_lo):
    sol_hi = solver_hi.solve_for_N02(N02)
    sol_lo = solver_lo.solve_for_N02(N02)

    sig_hi, vecs_hi = sol_hi["sigma"], sol_hi["vecs"]
    sig_lo, vecs_lo = sol_lo["sigma"], sol_lo["vecs"]

    # Frequency filtering
    matches = sp.match_frequencies(sig_hi, sig_lo)
    matches = sp.frequency_filter(sig_hi, sig_lo, matches, tol=rel_diff_tol)

    # Eigenfunction filtering
    filtered_modes = []
    # High and low resolution grids
    r_hi = sol_hi["r"]
    r_lo = sol_lo["r"]
    Nhi = sol_hi["N"]
    Nlo = sol_lo["N"]

    for i, j in matches:
        # Etas from high and low resolution
        eta_hi = vecs_hi[:, i]
        eta_lo = vecs_lo[:, j]
        # Eta1 and eta 2 from high and low resolution
        eta1_hi, eta2_hi = eta_hi[:Nhi], eta_hi[Nhi:]
        eta1_lo, eta2_lo = eta_lo[:Nlo], eta_lo[Nlo:]
        # NORMALIZATION (as eigenvs. are defined up to arbitr. ct. and sign
        eta1_hi = sp.normalize(eta1_hi, r_hi) # high res.: no interp needed
        eta2_hi = sp.normalize(eta2_hi, r_hi)
        # with interpolation  (_i)
        eta1_lo_i = sp.normalize(sp.interpolate_to_hi(r_hi, eta1_hi), r_hi)
        eta2_lo_i = sp.normalize(sp.interpolate_to_hi(r_hi, eta2_hi), r_hi)
        # To account for a possible sign ambiguity:
        if np.trapezoid(eta1_hi * eta2_hi, r_hi) < 0:
            eta2_hi *= -1
        if np.trapezoid(eta1_lo_i * eta2_lo_i, r_hi) < 0:
            eta2_lo_i *= -1
        # Mean square difference filtering
        if sp.mean_square_difference(eta1_hi, eta1_lo_i, r_hi) > mean_sq_threshold:
            continue
        if sp.mean_square_difference(eta2_hi, eta2_lo_i, r_hi) > mean_sq_threshold:
            continue
        # Number of nodes filtering
        if sp.count_nodes(eta1_hi) > mode_count_threshold:
            continue
        if sp.count_nodes(eta2_hi) > mode_count_threshold:
            continue
        if sp.count_nodes(eta1_lo) > mode_count_threshold:
            continue
        if sp.count_nodes(eta2_lo) > mode_count_threshold:
            continue

        filtered_modes.append({
            "sigma": sig_hi[i],
            "eta1": eta1_hi,
            "eta2": eta2_hi
        })

    return filtered_modes, sol_hi["alpha_R"]


####################################################################
if __name__=="__main__":

    N0_2_values = 10**np.linspace(-4, 1, num=100, endpoint=True)  # Dimitra's

    models = {
        "NS1": dict(Gamma1=2.0, Cs2=0.1, M=1.4*1.9884e30, R=1e4, alphaR=0.81),
        "NS2": dict(Gamma1=2.0, Cs2=0.1, M=2.0*1.9884e30, R=1e4, alphaR=0.74),
    }

    res_hi = dict(domains=2, points_per_domain=128)
    res_lo = dict(domains=3, points_per_domain=64)

    results = {}

    for name, pars in models.items():
        print(f"\n=== Processing {name} ===")

        solver_hi = ComputeToyModes(**pars, **res_hi)
        solver_lo = ComputeToyModes(**pars, **res_lo)

        results[name] = {}

        for N02 in N0_2_values:
            print(f"{name}: N0^2 = {N02:.3e}")

            modes, alpha_R = filtered_modes_for_N02(
                N02, solver_hi, solver_lo
            )

            results[name][N02] = {
                "alpha_R": alpha_R,
                "modes": modes
            }


    plt.figure(figsize=(5.4, 8))
    colors = {
        "NS1": "red",
        "NS2": "blue",
    }
    for name, data in results.items():
        x, y = [], []
        Cs2 = models[name]["Cs2"]

        for N02, entry in data.items():
            for m in entry["modes"]:
                x.append(N02)
                y.append(np.abs(m["sigma"]) /
                         (entry["alpha_R"] * np.sqrt(Cs2)))

        plt.scatter(x, y, color=colors[name], label=name, marker="*", alpha=0.7, s=1.2)
    # Analytical solutions for N_0² = 0 (Dimitra's)
    x_min, x_max = 1e-3, 10**(0.9)
    analytical_roots = np.array([5.7634591968945497914, 
                                 9.0950113304763551562, 
                                 12.322940970566582052,
                                 15.514603010886748230,
                                 18.689036355362822202,
                                 21.853874222709765792,
                                 25.012803202289612466,
                                 28.167829707993623875,
                                 31.320141707447174536,
                                 34.470488331284988666])
    plt.hlines(
        y=analytical_roots,
        xmin=x_min,
        xmax=x_max,
        colors='k',
        linestyles='-'
    )

    # Comparison with Dimitra's data
    """
    N0_2_dimitra = 10**np.linspace(-4, 1, num=100, endpoint=True)  # Dimitra's
    data_NS1 = np.loadtxt("/home/roque/Documents/phd/valencia/Data_test_case1-20260108T150451Z-1-001/Data_test_case1/data_NS1", comments="#")
    data_NS2 = np.loadtxt("/home/roque/Documents/phd/valencia/Data_test_case1-20260108T150451Z-1-001/Data_test_case1/data_NS2", comments="#")
    label_added = False
    size_data = 5
    for mode in data_NS1:
        if not label_added:
            plt.scatter(N0_2_dimitra, mode/(0.81 * np.sqrt(0.1)), marker="+", color="orange", alpha=0.4, s=size_data, label="NS1 (data)")
            label_added = True
        else:
            plt.scatter(N0_2_dimitra, mode/(0.81 * np.sqrt(0.1)), marker="+", color="orange", alpha=0.4, s=size_data)

    label_added = False
    for mode in data_NS2:
        if not label_added:
            plt.scatter(N0_2_dimitra, mode/(0.74 * np.sqrt(0.1)), marker="+", color="green", alpha=0.4, s=size_data, label="NS2 (data)")
            label_added = True
        else:
            plt.scatter(N0_2_dimitra, mode/(0.74 * np.sqrt(0.1)), marker="+", color="green", alpha=0.4, s=size_data)
    """
    # Similar plot adjustments as in FIGURE 2 in the article
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel(r"$N_0^2$")
    plt.ylabel(r"$\sigma / (\alpha_R c_s)$")
    plt.legend(loc="center left")
    plt.ylim(10**(-1.2), 10**(1.5))
    plt.xlim(x_min, x_max)
#    plt.show()
    plt.title("Freq. rel. diff.: %.2f ; MSD: %.2f ; # node thresh.: %i" % (rel_diff_tol, mean_sq_threshold, mode_count_threshold))
    # Resolve repo root from this script's location
    repository_root = Path(__file__).resolve().parents[1]

    # Target folder for plots
    plot_dir = repository_root / "data" / "output" / "mode_sols"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(plot_dir / "b.pdf")
