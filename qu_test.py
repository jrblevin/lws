#!/usr/bin/env python3
"""Qu (2011) test for true long memory against spurious long memory.

This implementation follows the original R code provided by Zhongjun Qu.

References
----------

- Qu, Z. (2011). A test against spurious long memory. _Journal of
  Business & Economic Statistics_ 29, 423--438.
"""

import numpy as np
from pyelw import LW


def qu_test(data, m=None, epsilon=0.05):
    """
    Qu (2011) test for true long memory against spurious long memory.

    Parameters
    ----------
    data : array_like
        Time series data of length n.
    m : int, optional
        Bandwidth parameter specifying the number of Fourier frequencies
        used for estimation. Default is round(n^0.7) following Qu's code.
    epsilon : float, optional
        Trimming parameter. Default is 0.05.
        For n > 500, epsilon=0.02 is recommended.

    Returns
    -------
    dict
        Dictionary containing:
        - 'W_stat': The test statistic (supremum of |cumsum| over trimmed range)
        - 'critical_values': Critical values for different significance levels
        - 'd_hat': Local Whittle estimate of d
        - 'reject_10': True if null rejected at 10% level
        - 'reject_05': True if null rejected at 5% level
        - 'reject_01': True if null rejected at 1% level

    Notes
    -----
    The null hypothesis is true long memory (fractional integration).
    The alternative is spurious long memory (e.g., due to structural breaks).
    This implementation follows Qu's original R code (RV5.R).
    Critical values are from Qu (2011) Table 1.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)

    # Demean data
    x = data - np.mean(data)

    # Default bandwidth: m = round(n^0.7)
    if m is None:
        m = round(n**0.7)
    m = min(m, n - 1)

    # Fourier frequencies
    freq = np.arange(1, n + 1) * (2 * np.pi / n)

    # Periodogram of demeaned data
    xf = np.fft.fft(x)
    px = (np.real(xf)**2 + np.imag(xf)**2) / (2 * np.pi * n)
    perdx = px[1:n]

    # Local Whittle estimation using PyELW
    d_hat = LW().fit(x, m=m).d_hat_
    # Convert to Qu's h parameterization: h = d + 0.5
    h_hat = d_hat + 0.5

    # Compute lambda_hat and G_hat at the optimum
    lambda_hat = freq[:m]**(2*h_hat - 1)
    G_hat = np.mean(perdx[:m] * lambda_hat)

    # Compute test statistic components
    comp1 = (perdx[:m] * lambda_hat) / G_hat
    comp2 = np.log(freq[:m]) - np.mean(np.log(freq[:m]))
    stat = np.cumsum((comp1 - 1) * comp2) / np.sqrt(np.sum(comp2**2))

    # Trimming
    trm = round(epsilon * m)
    trm = max(1, trm)  # Ensure at least 1

    # W statistic
    W_stat = np.max(np.abs(stat[trm-1:m]))

    # Critical values from Qu (2011) Table 1
    if epsilon <= 0.02:
        critical_values = {
            '10%': 1.118,
            '5%': 1.252,
            '2.5%': 1.374,
            '1%': 1.517
        }
    else:  # epsilon = 0.05
        critical_values = {
            '10%': 1.022,
            '5%': 1.155,
            '2.5%': 1.277,
            '1%': 1.426
        }

    return {
        'W_stat': W_stat,
        'critical_values': critical_values,
        'd_hat': d_hat,
        'h_hat': h_hat,
        'reject_10': W_stat > critical_values['10%'],
        'reject_05': W_stat > critical_values['5%'],
        'reject_01': W_stat > critical_values['1%']
    }
