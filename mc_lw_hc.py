#!/usr/bin/env python3
"""
Hurvich and Chen (2000) tapered local Whittle estimator Monte Carlo.

This is a focused replication of the HC estimator Monte Carlo
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
hc_estimates = np.zeros((mc_reps, len(d_list)))

# Results matrices: (bias, se, mse) for each d value
hc_results = np.zeros((len(d_list), 3))

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

print("Replication of Table 2 of Shimotsu and Phillips (2005) (HC Estimator)")
print(f"n={n}, m=n^{{0.65}}={m}, replications={mc_reps}")
print()
print("============================================================")
print("|     |        SP (2005)         |          PyELW          |")
print("|  d  |   bias    s.d.    MSE    |   bias    s.d.   MSE    |")
print("============================================================")

# Loop over experiments
for i, d_true in enumerate(d_list):
    hc_orig = original_hc[d_true]

    for rep in range(mc_reps):
        x = arfima(n, d_true, sigma=1.0, seed=42 * i + rep)
        hc_result = lw.estimate(x, m=m, taper='hc', bounds=(-4.0, 4.0), verbose=False)
        hc_estimates[rep, i] = hc_result['d_hat']

    # Calculate results for each d value
    hc_results[i, 0] = np.mean(hc_estimates[:, i]) - d_true  # Bias
    hc_results[i, 1] = np.std(hc_estimates[:, i])  # S.E.
    hc_results[i, 2] = np.mean((hc_estimates[:, i] - d_true)**2)  # MSE

    print(f"|{d_true:4.1f} |  {hc_orig[0]:7.4f} {hc_orig[1]:7.4f} {hc_orig[2]:7.4f} "
          f"| {hc_results[i, 0]:7.4f} {hc_results[i, 1]:7.4f} {hc_results[i, 2]:7.4f} |")

print("============================================================")

# Generate LaTeX table
latex_table = f"""\\begin{{table}}[t!]
\\centering
\\begin{{threeparttable}}
\\caption{{HC Tapered LW Estimator: Replication of Left Panel of Table 2 of Shimotsu and Phillips (2005)}}
\\label{{tab:mc:lw_hc}}
\\begin{{tabular}}{{r@{{\\hspace{{1em}}}}rrr@{{\\hspace{{1em}}}}rrr}}
\\toprule
\\multicolumn{{1}}{{c}}{{}} & \\multicolumn{{3}}{{c}}{{Original}} & \\multicolumn{{3}}{{c}}{{Replication}} \\\\
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}}
$d$ & Bias & S.D. & MSE & Bias & S.D. & MSE \\\\
\\midrule
"""

for i, d_true in enumerate(d_list):
    hc_orig = original_hc[d_true]
    latex_table += f"${d_true:4.1f}$ & ${hc_orig[0]:7.4f}$ & ${hc_orig[1]:7.4f}$ & ${hc_orig[2]:7.4f}$ & ${hc_results[i, 0]:7.4f}$ & ${hc_results[i, 1]:7.4f}$ & ${hc_results[i, 2]:7.4f}$ \\\\\n"

latex_table += f"""\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\footnotesize
\\item Notes: Hurvich-Chen tapered LW estimator, $n = {n}$ observations from $\\ARFIMA(0,d,0)$, $m = \\lfloor n^{{{alpha:.2f}}} \\rfloor = {m}$ frequencies, {mc_reps:,} replications.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
"""

# Save LaTeX table
with open('tables/mc_lw_hc.tex', 'w') as f:
    f.write(latex_table)

print("LaTeX table saved to: tables/mc_lw_hc.tex")
