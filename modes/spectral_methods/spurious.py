import numpy as np
from scipy.interpolate import interp1d


def clean_and_sort(sigma, vecs, sigma_min):
    mask = np.abs(sigma) > sigma_min
    sigma = sigma[mask]
    vecs = vecs[:, mask]

    idx = np.argsort(np.abs(sigma))
    return sigma[idx], vecs[:, idx]


def match_frequencies(sig_hi, sig_lo):
    matches = []
    for i, s in enumerate(sig_hi):
        j = np.argmin(np.abs(sig_lo - s))
        matches.append((i, j))
    return matches


def frequency_filter(sig_hi, sig_lo, matches, tol):
    kept = []
    for i, j in matches:
        rel = np.abs(sig_hi[i] - sig_lo[j]) / np.abs(sig_hi[i])
        if rel < tol:
            kept.append((i, j))
    return kept


def interpolate_to_hi(r_hi, f_hi):
    f = interp1d(r_hi, f_hi, kind="cubic", fill_value="extrapolate")
    return f(r_hi)


def normalize(f, r_common):
    return f / np.sqrt(np.trapezoid(np.abs(f)**2, r_common))


def mean_square_difference(f1, f2, x_common):
    return np.trapezoid((f1 - f2)**2, x_common) # Divide by R=1 ?


def count_nodes(f):
    f = np.real(f)
    return np.sum(f[:-1] * f[1:] < 0)
