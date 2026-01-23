import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from .spectral import AB_2x2_system



class ToyModelsTseneklidou:
    
    """
    This class defines the 'toy models' NS1 and NS2 
    from section V (A. Test case 1: p and g-modes) in 
    https://doi.org/10.1103/b3ty-cr5g
    """
    
    def __init__(self, toy_params, spectral_grid):
        self.toy_params = toy_params
        self.spectral_grid = spectral_grid

    def __call__(self):
        
        # Spectral grid
        N   = self.spectral_grid.N
        D  = self.spectral_grid.D
        r   = self.spectral_grid.r
    
        # Toy model parameters (include as input when using the class)
        Cs2    = self.toy_params['Cs2']
        N0_2   = self.toy_params['N2_0']
        Gamma1 = self.toy_params['Gamma1']
        M      = self.toy_params['M_kg']    # kg
        alpha_R = self.toy_params['alpha_R']
        
        # Brunt-Väisälä frequency
        r_min = 0.5
        N2 = (N0_2 / 4.0) * ((np.tanh(10.0 * (r - r_min)) + 1.0)**2)
    
        # alpha'(r) = +R_alpha^{-1} (derivative of lapse function)
        alpha_prime = np.sqrt((N2 * Cs2) / (Gamma1 - 1.0 - Cs2))  # plus sign
        self.alpha_R = alpha_R   # If needed later (e.g. plot in FIG 2)
        
        # Integrate alpha'(r)    
        int_Ra_0_r = cumulative_trapezoid(alpha_prime[:,0], 
                                          r[:,0],
                                          initial=0.0)
        int_Ra_0_1 = trapezoid(alpha_prime[:,0], r[:,0])
        alpha = (alpha_R - (int_Ra_0_1 - int_Ra_0_r)).reshape(N,1)
        
        # Gravitational acceleration
        Gp = -alpha_prime/alpha  # = dp_dr / (rho * h)
        
        # Integral for rho(r) (density)
        integrand = (alpha[:,0]**(-Gamma1 / Cs2)) * (r[:,0]**2)
        denom = trapezoid(integrand, r[:,0])    
        # rho(r): shape (N,1)
        rho = (M/ (4.0 * np.pi)) * (alpha**(-Gamma1 / Cs2)) / denom

        # Enthalpy h
        h = (Gamma1 - 1.0) / (Gamma1 - 1.0 - Cs2) 
        # dh/dr
        dh_dr = 0.
        
        # p(r) (pressure; computed using the adiabatic condition)
        p = ((rho * h * Cs2) / Gamma1)
        
        # Derivatives of rho and p
        drho_dr = D @ rho
        dp_dr = ((h * Cs2) / Gamma1) * drho_dr
        
        # psi(r) in the metric
        psi = 1.0  # flat
        # dpsi/dr
        dpsi_dr = 0.0
        
        # Relativistic Schwarzschild discriminant
        Brel = drho_dr / rho - dp_dr / (rho*h) - dp_dr / (Gamma1*p)

        # mode "quantum" number l
        l = 2
        
        # Define A and B from the 2x2 system A eta = sigma^2 B
        A, B = AB_2x2_system(l, r, D, alpha, psi, dpsi_dr, Brel, Gp, rho, 
                             drho_dr, h, dh_dr, Cs2)
        
        ########################
        # BOUNDARY CONDITIONS at r=R: IMPOSE eta_2(r=R)=0: see eq. (90)
        A[2*N-1, :]     = 0.0
        B[2*N-1, :]     = 0.0
        A[2*N-1, 2*N-1] = 1.0 
        #######################
        
        return A, B





