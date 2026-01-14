import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from spectral import SpectralGrid
from ns_models_2x2_system import ToyModelsTseneklidou



class ComputeToyModes:
    def __init__(self, Gamma1, Cs2, M, R, alphaR, points_per_domain, domains, r_shock=1.):
        self.Gamma1 = Gamma1
        self.Cs2 = Cs2
        self.M = M
        self.R = R
        self.alphaR = alphaR
        self.domains = domains
        self.points_per_domain = points_per_domain
        self.grid = SpectralGrid(r_shock, points_per_domain, domains)

        
    def solve_for_N02(self, N02):
        
        # Toy model parameters
        toy_params = {
            "Cs2": self.Cs2,
            "N2_0": N02,
            "Gamma1": self.Gamma1,
            "M_kg": self.M,
            "R_m": self.R,
            "alpha_R": self.alphaR
        }

        # Get matrices A and B
        eq = ToyModelsTseneklidou(toy_params, self.grid)
        A, B = eq()
        
        # Solve eigenvalue problem
        lam, _ = eig(A, B)
        sigma = np.sqrt(lam)

        # x,y: to plot (see Figure 2 in the article)
        cs = np.sqrt(self.Cs2)
        alpha_R = eq.alpha_R
        y = sigma / (alpha_R * cs)
        x = np.full_like(y, N02)

        return x, y
    

    # Solve for different values of N_0^2
    def scan(self, N02_values):
        all_x, all_y = [], []
        for N02 in N02_values:
            print(f"Solving N0^2 = {N02}")
            x, y = self.solve_for_N02(N02)
            all_x.append(x)
            all_y.append(y)
        return np.concatenate(all_x), np.concatenate(all_y)




# N_0 values
N0_2_values = 10**np.linspace(-4, 1, num=100, endpoint=True)  # Dimitra's
#N0_2_values = np.logspace(-3, 1, 50)  # equally spaced over an interval in log

# spectral collocation parameters
points_per_domain = 64
domains = 1

# NS1 
Gamma1_NS1 = 2.0           # Adiabatic index
Cs2_NS1    = 0.1           # Sound speed squared
M_NS1      = 1.4 * 1.9884e30  # Mass in kg
R_NS1      = 1e4           # Radius in meters
alphaR_NS1 = 0.81          # Lapse function at radius 
scanner_NS1 = ComputeToyModes(Gamma1=Gamma1_NS1, 
                             Cs2=Cs2_NS1, 
                             M=M_NS1, 
                             R=R_NS1, 
                             alphaR=alphaR_NS1,
                             points_per_domain=points_per_domain, 
                             domains=domains)
x_NS1, y_NS1 = scanner_NS1.scan(N0_2_values)

# NS2
Gamma1_NS2 = 2.0           # Adiabatic index
Cs2_NS2    = 0.1           # Sound speed squared
M_NS2      = 2.0 * 1.9884e30  # Mass in kg
R_NS2      = 1e4         # Radius in meters
alphaR_NS2 = 0.74          # Lapse function at radius 
scanner_NS2 = ComputeToyModes(Gamma1=Gamma1_NS2, 
                             Cs2=Cs2_NS2, 
                             M=M_NS2, 
                             R=R_NS2, 
                             alphaR=alphaR_NS2,
                             points_per_domain=points_per_domain, 
                             domains=domains)
x_NS2, y_NS2 = scanner_NS2.scan(N0_2_values)



# Plot of eigenfrequencies versus N_{0}^{2} (see FIGURE 2 in the article)

plt.figure(figsize=(5.4,8))
plt.scatter(x_NS1, y_NS1, s=3, color="red", label="NS1")
plt.scatter(x_NS2, y_NS2, s=3, color="blue", label="NS2")

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

# Similar plot adjustments as in FIGURE 2 in the article
plt.xscale('log')
plt.yscale('log')

plt.xlabel(r"$N_0^2$")
plt.ylabel(r"$\sigma / (\alpha_R c_s)$")

plt.ylim(10**(-1.2), 10**(1.5))
plt.xlim(x_min, x_max)

plt.title("Domains: %i Points per domain: %i" % (points_per_domain, domains))
plt.legend(loc="center left")

plt.grid(0)#(True, which="both", ls="--", lw=0.5)
plt.show()