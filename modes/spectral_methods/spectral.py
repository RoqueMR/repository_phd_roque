import numpy as np



class SpectralGrid:
    """
    Credit (for this class): Dimitra Tseneklidou
    (So far, consider r_shock=1 only: zero velocity limit)
    Use as:
    SpectralGrid(r_shock, points_per_domain, domains)
    """
    
    def __init__(self, r_shock, points_per_domain, domains):
        D, r = self.multicheb(points_per_domain, domains)
        self.r = - r_shock * (r - 1.0) / 2.0
        self.D = - 2 * D / r_shock
        self.N = self.D.shape[0]
        self.r_shock = r_shock
        self.points_per_domain = points_per_domain
        self.domains = domains

    def cheb(self, N):
            r = np.arange(0, N + 1)
            x = (np.cos(np.pi * r / N)).reshape(N + 1, 1)
            c = (np.concatenate([[2], np.ones((N - 1)), [2]]) * (-1) ** r).reshape(N + 1, 1)
            X = np.ones((1, N + 1)) * x
            dX = X - np.transpose(X)
            D = np.matmul(c, np.transpose(1 / c)) / (dX + np.identity(N + 1))
            D = D - np.diag(np.sum(D, axis=1))
            return D, x

    def multicheb(self, N, M):
            total_size = M * N + 1
            dx = 2.0 / M
            D, x = self.cheb(N)
            bigD = np.zeros((total_size, total_size))
            bigx = np.zeros((total_size, 1))
            for m in range(M):
                bigx[m * N:(N + 1) + m * N, 0] = (x[:, 0] - 1) / M - dx * m + 1
                bigD[m * N:(N + 1) + m * N, m * N:(N + 1) + m * N] += D * M
                if m > 0:
                    bigD[m * N, :] /= 2
            return bigD, bigx
        


def AB_2x2_system(l, r, D, alpha, psi, dpsi_dr, Brel, Gp, rho, drho_dr, h, dh_dr, Cs2):
    """
    Specifies the 2x2 system from equation (47) in 
    https://doi.org/10.1103/b3ty-cr5g
    """
    
    N = len(r)
    A = np.zeros((2*N, 2*N), dtype=complex)
    B = np.zeros((2*N, 2*N), dtype=complex)

    I = np.identity(N)
    r_matrix = np.diag(r[:,0])

    A[:N, :N]       = np.diag((((alpha**2) * Brel * Gp) / (psi**4))[:,0])
    A[:N, N:2*N]    = 0.0 * I
    A[N:2*N, :N]    = -r_matrix@D - 2.0*I - \
                      np.diag((r*Gp/Cs2 + 6.0*r*dpsi_dr/psi)[:,0])
    A[N:2*N, N:2*N] = l*(l + 1.0) * I

    B[:N, :N]       = 1.0 * I
    B[:N, N:2*N]    = -r_matrix@D - 1.0*I - np.diag((r*Gp*(1.0 - 1.0/Cs2) + \
                      r*(drho_dr/rho + dh_dr/h + 4.0*dpsi_dr/psi))[:,0])
    B[N:2*N, :N]    = 0.0 * I
    B[N:2*N, N:2*N] = np.diag((((r**2) * (psi**4)) / ((alpha**2) * Cs2))[:,0])
    
    return A, B

