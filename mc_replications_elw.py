#!/usr/bin/env python3
"""
Exact local Whittle estimator replication Monte Carlo.

Replication of the ELW Monte Carlo results in the left panel of
Table 1 of Shimotsu and Phillips (2005) and the two-step ELW Monte
Carlo results in Table 2 of Shimotsu (2010), combined into a single
two-panel table.
"""

import numpy as np

from pyelw import ELW, TwoStepELW
from pyelw.simulate import arfima

from common import write_replication_table

# Settings for Panel A (ELW), from Shimotsu and Phillips (2005) Table 1
n = 500
d_list = [-3.5, -2.3, -1.7, -1.3, -0.7, -0.3, 0.0, 0.3, 0.7, 1.3, 1.7, 2.3, 3.5]
mc_reps = 10000
alpha = 0.65
m = int(n**alpha)

# Original ELW results from Table 1 of Shimotsu and Phillips (2005)
original_elw = {
    -3.5: (-0.0024, 0.0787, 0.0062),
    -2.3: (-0.0020, 0.0774, 0.0060),
    -1.7: (-0.0020, 0.0776, 0.0060),
    -1.3: (-0.0014, 0.0770, 0.0059),
    -0.7: (-0.0024, 0.0787, 0.0062),
    -0.3: (-0.0033, 0.0777, 0.0060),
    0.0: (-0.0029, 0.0784, 0.0061),
    0.3: (-0.0020, 0.0782, 0.0061),
    0.7: (-0.0017, 0.0777, 0.0060),
    1.3: (-0.0014, 0.0781, 0.0061),
    1.7: (-0.0025, 0.0780, 0.0061),
    2.3: (-0.0026, 0.0772, 0.0060),
    3.5: (-0.0016, 0.0770, 0.0059),
}

# Panel A: ELW estimator, left panel of Table 1 of Shimotsu and Phillips (2005)

elw = ELW()
elw_rows = []

print("Replication of Table 1 of Shimotsu and Phillips (2005) (Left Panel)")
print(f"n={n}, m=n^{{0.65}}={m}, replications={mc_reps}")
print()
print("============================================================")
print("|     |        SP (2005)         |          PyELW          |")
print("|  d  |   bias    s.d.    MSE    |   bias    s.d.   MSE    |")
print("============================================================")

elw_estimates = np.zeros(mc_reps)
for i, d_true in enumerate(d_list):
    elw_orig = original_elw[d_true]

    for rep in range(mc_reps):
        x = arfima(n, d_true, sigma=1.0, seed=42 * i + rep)
        elw_result = elw.estimate(x, m=m, bounds=(-4.0, 4.0), verbose=False)
        elw_estimates[rep] = elw_result['d_hat']

    elw_bias = np.mean(elw_estimates) - d_true
    elw_sd = np.std(elw_estimates)
    elw_mse = np.mean((elw_estimates - d_true)**2)
    elw_rows.append((d_true, *elw_orig, elw_bias, elw_sd, elw_mse))

    print(f"|{d_true:4.1f} |  {elw_orig[0]:7.4f} {elw_orig[1]:7.4f} {elw_orig[2]:7.4f} "
          f"| {elw_bias:7.4f} {elw_sd:7.4f} {elw_mse:7.4f} |")

print("============================================================")
print()

# Panel B: 2ELW estimator, Table 2 of Shimotsu (2010)

# Settings from Shimotsu (2010) Table 2
n2 = 512
d_list_2elw = [0.0, 0.4, 0.8, 1.2]
rho_list = [0.0, 0.5, 0.8]
m2 = int(n2**alpha)  # m = n^0.65 = 57

print(f"Monte Carlo replication: n={n2}, m=n^{{{alpha:.2f}}}={m2}, replications={mc_reps}")

# Two-step ELW estimator
elw2 = TwoStepELW(bounds=(-1.0, 3.0), trend_order=0)

# Store results for the combined table
elw2_rows = []

# Original results from Shimotsu (2010) Table 2 (2ELW columns only)
original_2elw = {
    (0.0, 0.0): (-0.0022, 0.0058),
    (0.0, 0.5): (0.0994, 0.0061),
    (0.0, 0.8): (0.4133, 0.0072),
    (0.4, 0.0): (0.0001, 0.0058),
    (0.4, 0.5): (0.1003, 0.0060),
    (0.4, 0.8): (0.4160, 0.0072),
    (0.8, 0.0): (-0.0003, 0.0058),
    (0.8, 0.5): (0.0988, 0.0060),
    (0.8, 0.8): (0.4125, 0.0073),
    (1.2, 0.0): (-0.0006, 0.0057),
    (1.2, 0.5): (0.0990, 0.0061),
    (1.2, 0.8): (0.4117, 0.0070)
}

print("===============================================================================")
print("|     |     | Original |  PyELW   |          | Original |   PyELW  |          |")
print("|  d  | rho |   Bias   |   Bias   |   Diff   |   Var    |    Var   |   Diff   |")
print("===============================================================================")

# Loop over all parameter combinations
for d_true in d_list_2elw:
    for rho in rho_list:
        estimates = np.zeros(mc_reps)

        # Monte Carlo simulation
        for rep in range(mc_reps):
            # Generate ARFIMA(1,d,0) process
            seed = (42 * len(d_list_2elw) * len(rho_list) * rep
                    + d_list_2elw.index(d_true) * len(rho_list) + rho_list.index(rho))
            x = arfima(n2, d_true, phi=rho, sigma=1.0, seed=seed, burnin=2*n2)

            # Apply 2-step ELW estimator
            result = elw2.estimate(x, m=m2)
            estimates[rep] = result['d_hat']

        # Calculate bias and variance
        pyelw_bias = np.mean(estimates) - d_true
        pyelw_var = np.var(estimates)

        # Get original results
        orig_bias, orig_var = original_2elw[(d_true, rho)]

        # Calculate differences
        bias_diff = pyelw_bias - orig_bias
        var_diff = pyelw_var - orig_var

        # Store results for the combined table (only rho=0 is reported).
        # Shimotsu (2010) reports bias and variance, so the original S.D.
        # and MSE are computed as sqrt(variance) and variance + bias^2.
        if rho == 0.0:
            elw2_rows.append((d_true, orig_bias, np.sqrt(orig_var),
                              orig_var + orig_bias**2, pyelw_bias,
                              np.std(estimates),
                              np.mean((estimates - d_true)**2)))

        print(f"| {d_true:3.1f} | {rho:3.1f} | {orig_bias:8.4f} | {pyelw_bias:8.4f} | {bias_diff:8.4f} |"
              f" {orig_var:8.4f} | {pyelw_var:8.4f} | {var_diff:8.4f} |")

print("===============================================================================")
print()

# Combine the two panels into Table 2 of the paper
write_replication_table(
    'tables/mc_replications_elw.tex',
    caption=('Replication of Published Monte Carlo Results: '
             'Exact Local Whittle Estimators'),
    label='tab:mc:replications:elw',
    panels=[
        ('Panel A: Exact local Whittle (ELW)', elw_rows),
        ('Panel B: Two-step exact local Whittle (2ELW)', elw2_rows),
    ],
    notes=(
        'Bias, standard deviation, and mean squared error of the '
        'estimates over 10,000 replications of $\\ARFIMA(0,d,0)$. '
        "``Original'' columns report published values; ``Replication'' "
        'columns report results from our Python implementation. '
        'Panel A replicates the left panel of Table 1 of Shimotsu and '
        'Phillips (2005), with $n = 500$ and $m = 56$. '
        'Panel B replicates Table 2 of Shimotsu (2010), with $n = 512$ '
        'and $m = 57$.'
    ),
)
