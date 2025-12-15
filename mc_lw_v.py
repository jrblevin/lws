#!/usr/bin/env python3
"""
Velasco (1999) tapered Local Whittle estimator Monte Carlo.

This is a replication of the 'V' estimator Monte Carlo
from Table 2 of Shimotsu and Phillips (2005).
"""

import numpy as np

from pyelw import LW
from pyelw.simulate import arfima

# Settings
n = 500
d_list = [-3.5, -2.3, -1.7, -1.3, -0.7, -0.3, 0.0, 0.3, 0.7, 1.3, 1.7, 2.3, 3.5]
mc_reps = 10000
alpha = 0.65
m = int(n**alpha)

# Estimators
lw = LW()

# Initialize storage for results
v_estimates = np.zeros((mc_reps,))
results_list = []  # For LaTeX generation

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

print("Replication of Table 2 of Shimotsu and Phillips (2005) (V Estimator)")
print(f"n={n}, m=n^{{0.65}}={m}, replications={mc_reps}, Bartlett taper")
print()
print("============================================================")
print("|     |        SP (2005)         |          PyELW          |")
print("|  d  |   Bias    S.D.    MSE    |   Bias    S.D.   MSE    |")
print("============================================================")

# Loop over experiments
for d_true in d_list:
    v_orig = original_v[d_true]
    for rep in range(mc_reps):
        x = arfima(n, d_true, sigma=1.0, seed=42 + rep)
        v_result = lw.estimate(x, m=m, taper='bartlett', bounds=(-4.0, 4.0), verbose=False)
        v_estimates[rep] = v_result['d_hat']

    # Calculate results for each d value
    v_bias = np.mean(v_estimates) - d_true
    v_sd = np.std(v_estimates)
    v_mse = np.mean((v_estimates - d_true)**2)

    # Store for LaTeX generation
    results_list.append((d_true, v_orig, v_bias, v_sd, v_mse))

    print(f"|{d_true:4.1f} |  {v_orig[0]:7.4f} {v_orig[1]:7.4f} {v_orig[2]:7.4f} "
          f"| {v_bias:7.4f} {v_sd:7.4f} {v_mse:7.4f} |")

print("============================================================")

# Generate LaTeX table
latex_table = f"""\\begin{{table}}[tbp]
\\centering
\\begin{{threeparttable}}
\\caption{{Velasco Tapered LW Estimator: Replication of Right Panel of Table 2 of Shimotsu and Phillips (2005)}}
\\label{{tab:mc:lw_v}}
\\begin{{tabular}}{{r@{{\\hspace{{1em}}}}rrr@{{\\hspace{{1em}}}}rrr}}
\\toprule
\\multicolumn{{1}}{{c}}{{}} & \\multicolumn{{3}}{{c}}{{Original}} & \\multicolumn{{3}}{{c}}{{Replication}} \\\\
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}}
$d$ & Bias & S.D. & MSE & Bias & S.D. & MSE \\\\
\\midrule
"""

for d_true, v_orig, v_bias, v_sd, v_mse in results_list:
    latex_table += f"${d_true:4.1f}$ & ${v_orig[0]:7.4f}$ & ${v_orig[1]:7.4f}$ & ${v_orig[2]:7.4f}$ & ${v_bias:7.4f}$ & ${v_sd:7.4f}$ & ${v_mse:7.4f}$ \\\\\n"

latex_table += f"""\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\footnotesize
\\item Notes: Velasco tapered LW estimator with Bartlett taper, $n = {n}$ observations from $\\ARFIMA(0,d,0)$, $m = \\lfloor n^{{{alpha:.2f}}} \\rfloor = {m}$ frequencies, {mc_reps:,} replications.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
"""

# Save LaTeX table
with open('tables/mc_lw_v.tex', 'w') as f:
    f.write(latex_table)

print("LaTeX table saved to: tables/mc_lw_v.tex")
