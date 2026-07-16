#!/usr/bin/env python3
"""
LW and tapered LW estimator replication Monte Carlo.

Replication of the Monte Carlo results for the LW, Velasco (Bartlett
taper), and Hurvich-Chen tapered estimators from Tables 1 and 2 of
Shimotsu and Phillips (2005), combined into a single three-panel table.
"""

import numpy as np

from pyelw import LW
from pyelw.simulate import arfima

from common import write_replication_table

# Settings
n = 500
d_list_lw = [-1.3, -0.7, -0.3, 0.0, 0.3, 0.7, 1.3]
d_list = [-3.5, -2.3, -1.7, -1.3, -0.7, -0.3, 0.0, 0.3, 0.7, 1.3, 1.7, 2.3, 3.5]
mc_reps = 10000
alpha = 0.65
m = int(n**alpha)

# Estimator (the Velasco and HC variants are selected via the taper option)
lw = LW()

# Original LW results from Table 1 of Shimotsu and Phillips (2005)
original_lw = {
    -3.5: (3.1617, 0.2831, 10.076),
    -2.3: (1.6345, 0.3041, 2.7640),
    -1.7: (0.8709, 0.2788, 0.8363),
    -1.3: (0.4109, 0.2170, 0.2160),
    -0.7: (0.0353, 0.0885, 0.0091),
    -0.3: (-0.0027, 0.0781, 0.0061),
    0.0: (-0.0075, 0.0781, 0.0062),
    0.3: (-0.0066, 0.0785, 0.0062),
    0.7: (0.0099, 0.0812, 0.0067),
    1.3: (-0.2108, 0.0982, 0.0541),
    1.7: (-0.6288, 0.1331, 0.4130),
    2.3: (-1.2647, 0.1046, 1.6104),
    3.5: (-2.4919, 0.0724, 6.2150)
}

# Original Velasco results from Table 2 of Shimotsu and Phillips (2005)
original_v = {
    -3.5: (1.6126, 0.3380, 2.7148),
    -2.3: (0.2155, 0.1726, 0.0762),
    -1.7: (0.0259, 0.1235, 0.0159),
    -1.3: (0.0081, 0.1211, 0.0147),
    -0.7: (-0.0068, 0.1219, 0.0149),
    -0.3: (-0.0133, 0.1224, 0.0151),
    0.0: (-0.0138, 0.1224, 0.0152),
    0.3: (-0.0132, 0.1235, 0.0154),
    0.7: (-0.0068, 0.1227, 0.0151),
    1.3: (0.0140, 0.1232, 0.0154),
    1.7: (0.0456, 0.1288, 0.0187),
    2.3: (-0.1781, 0.1419, 0.0519),
    3.5: (-1.4541, 0.1338, 2.1322)
}

# Original HC (Hurvich-Chen) results from Table 2 of Shimotsu and Phillips (2005)
original_hc = {
    -3.5: (2.5889, 0.3037, 6.7946),
    -2.3: (1.1100, 0.2893, 1.3157),
    -1.7: (0.4474, 0.2154, 0.2466),
    -1.3: (0.1551, 0.1231, 0.0392),
    -0.7: (0.0278, 0.0957, 0.0099),
    -0.3: (0.0100, 0.0971, 0.0095),
    0.0: (0.0034, 0.0985, 0.0097),
    0.3: (-0.0033, 0.1004, 0.0101),
    0.7: (-0.0066, 0.0994, 0.0099),
    1.3: (-0.0079, 0.0987, 0.0098),
    1.7: (0.0008, 0.0972, 0.0095),
    2.3: (0.0528, 0.0981, 0.0124),
    3.5: (-0.4079, 0.1142, 0.1795)
}

# Panel A: LW estimator, right panel of Table 1 of Shimotsu and Phillips (2005)

lw_rows = []

print("Replication of Table 1 of Shimotsu and Phillips (2005) (Right Panel)")
print(f"n={n}, m=n^{{0.65}}={m}, replications={mc_reps}")
print()
print("============================================================")
print("|     |        SP (2005)         |          PyELW          |")
print("|  d  |   bias    s.d.    MSE    |   bias    s.d.   MSE    |")
print("============================================================")

lw_estimates = np.zeros(mc_reps)
for i, d_true in enumerate(d_list_lw):
    lw_orig = original_lw[d_true]

    for rep in range(mc_reps):
        x = arfima(n, d_true, sigma=1.0, seed=42 * i + rep)
        lw_result = lw.estimate(x, m=m, bounds=(-4.0, 4.0), verbose=False)
        lw_estimates[rep] = lw_result['d_hat']

    lw_bias = np.mean(lw_estimates) - d_true
    lw_sd = np.std(lw_estimates)
    lw_mse = np.mean((lw_estimates - d_true)**2)
    lw_rows.append((d_true, *lw_orig, lw_bias, lw_sd, lw_mse))

    print(f"|{d_true:4.1f} |  {lw_orig[0]:7.4f} {lw_orig[1]:7.4f} {lw_orig[2]:7.4f} "
          f"| {lw_bias:7.4f} {lw_sd:7.4f} {lw_mse:7.4f} |")

print("============================================================")
print()

# Panel B: Velasco tapered LW, right panel of Table 2 of Shimotsu and
# Phillips (2005)

v_rows = []

print("Replication of Table 2 of Shimotsu and Phillips (2005) (V Estimator)")
print(f"n={n}, m=n^{{0.65}}={m}, replications={mc_reps}, Bartlett taper")
print()
print("============================================================")
print("|     |        SP (2005)         |          PyELW          |")
print("|  d  |   Bias    S.D.    MSE    |   Bias    S.D.   MSE    |")
print("============================================================")

v_estimates = np.zeros(mc_reps)
for d_true in d_list:
    v_orig = original_v[d_true]

    for rep in range(mc_reps):
        x = arfima(n, d_true, sigma=1.0, seed=42 + rep)
        v_result = lw.estimate(x, m=m, taper='bartlett', bounds=(-4.0, 4.0), verbose=False)
        v_estimates[rep] = v_result['d_hat']

    v_bias = np.mean(v_estimates) - d_true
    v_sd = np.std(v_estimates)
    v_mse = np.mean((v_estimates - d_true)**2)
    v_rows.append((d_true, *v_orig, v_bias, v_sd, v_mse))

    print(f"|{d_true:4.1f} |  {v_orig[0]:7.4f} {v_orig[1]:7.4f} {v_orig[2]:7.4f} "
          f"| {v_bias:7.4f} {v_sd:7.4f} {v_mse:7.4f} |")

print("============================================================")
print()

# Panel C: HC tapered LW, left panel of Table 2 of Shimotsu and Phillips (2005)

hc_rows = []

print("Replication of Table 2 of Shimotsu and Phillips (2005) (HC Estimator)")
print(f"n={n}, m=n^{{0.65}}={m}, replications={mc_reps}")
print()
print("============================================================")
print("|     |        SP (2005)         |          PyELW          |")
print("|  d  |   bias    s.d.    MSE    |   bias    s.d.   MSE    |")
print("============================================================")

hc_estimates = np.zeros(mc_reps)
for i, d_true in enumerate(d_list):
    hc_orig = original_hc[d_true]

    for rep in range(mc_reps):
        x = arfima(n, d_true, sigma=1.0, seed=42 * i + rep)
        hc_result = lw.estimate(x, m=m, taper='hc', bounds=(-4.0, 4.0), verbose=False)
        hc_estimates[rep] = hc_result['d_hat']

    hc_bias = np.mean(hc_estimates) - d_true
    hc_sd = np.std(hc_estimates)
    hc_mse = np.mean((hc_estimates - d_true)**2)
    hc_rows.append((d_true, *hc_orig, hc_bias, hc_sd, hc_mse))

    print(f"|{d_true:4.1f} |  {hc_orig[0]:7.4f} {hc_orig[1]:7.4f} {hc_orig[2]:7.4f} "
          f"| {hc_bias:7.4f} {hc_sd:7.4f} {hc_mse:7.4f} |")

print("============================================================")
print()

# Combine the three panels into Table 1 of the paper
write_replication_table(
    'tables/mc_replications_lw.tex',
    caption=('Replication of Published Monte Carlo Results: '
             'LW and Tapered LW Estimators'),
    label='tab:mc:replications:lw',
    panels=[
        ('Panel A: Local Whittle (LW)', lw_rows),
        ('Panel B: Velasco tapered LW, Bartlett taper (V)', v_rows),
        ('Panel C: Hurvich--Chen tapered LW (HC)', hc_rows),
    ],
    notes=(
        'Bias, standard deviation, and mean squared error of the '
        'estimates over 10,000 replications, each with $n = 500$ '
        'observations from $\\ARFIMA(0,d,0)$ and '
        '$m = \\lfloor n^{0.65} \\rfloor = 56$ frequencies. '
        "``Original'' columns report published values: Panel A from "
        'the right panel of Table 1 of Shimotsu and Phillips (2005), '
        'and Panels B and C from the right and left panels of their '
        "Table 2. ``Replication'' columns report results from our "
        'Python implementation.'
    ),
)
