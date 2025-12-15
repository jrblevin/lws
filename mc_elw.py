#!/usr/bin/env python3
"""
Exact Local Whittle estimator Monte Carlo

This is a focused replication of the left panel ELW Monte Carlo in Table 1 of
Shimotsu and Phillips (2005).
"""

import numpy as np

from pyelw import ELW
from pyelw.simulate import arfima

# Settings
n = 500
d_list = [-3.5, -2.3, -1.7, -1.3, -0.7, -0.3, 0.0, 0.3, 0.7, 1.3, 1.7, 2.3, 3.5]
mc_reps = 10000
alpha = 0.65
m = int(n**alpha)

# Estimators
elw = ELW()

# Initialize storage for results
estimates = np.zeros((mc_reps, len(d_list)))

# Results matrices: (bias, se, mse) for each d value
results = np.zeros((len(d_list), 3))

# Original ELW results from Table 1 of Shimotsu and Phillips (2005)
original = {
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

print("Replication of Table 1 of Shimotsu and Phillips (2005) (Left Panel)")
print(f"n={n}, m=n^{{0.65}}={m}, replications={mc_reps}")
print()
print("============================================================")
print("|     |        SP (2005)         |          PyELW          |")
print("|  d  |   bias    s.d.    MSE    |   bias    s.d.   MSE    |")
print("============================================================")

# Loop over experiments
for i, d_true in enumerate(d_list):
    lw_orig = original[d_true]

    for rep in range(mc_reps):
        x = arfima(n, d_true, sigma=1.0, seed=42 * i + rep)
        lw_result = elw.estimate(x, m=m, bounds=(-4.0, 4.0), verbose=False)
        estimates[rep, i] = lw_result['d_hat']

    # Calculate results for each d value
    results[i, 0] = np.mean(estimates[:, i]) - d_true  # Bias
    results[i, 1] = np.std(estimates[:, i])  # S.E.
    results[i, 2] = np.mean((estimates[:, i] - d_true)**2)  # MSE

    print(f"|{d_true:4.1f} |  {lw_orig[0]:7.4f} {lw_orig[1]:7.4f} {lw_orig[2]:7.4f} "
          f"| {results[i, 0]:7.4f} {results[i, 1]:7.4f} {results[i, 2]:7.4f} |")

print("============================================================")

# Generate LaTeX table
latex_table = f"""\\begin{{table}}[t!]
\\centering
\\begin{{threeparttable}}
\\caption{{ELW Estimator: Replication of Left Panel of Table 1 of Shimotsu and Phillips (2005)}}
\\label{{tab:mc:elw}}
\\begin{{tabular}}{{r@{{\\hspace{{1em}}}}rrr@{{\\hspace{{1em}}}}rrr}}
\\toprule
\\multicolumn{{1}}{{c}}{{}} & \\multicolumn{{3}}{{c}}{{Original}} & \\multicolumn{{3}}{{c}}{{Replication}} \\\\
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}}
$d$ & Bias & S.D. & MSE & Bias & S.D. & MSE \\\\
\\midrule
"""

for i, d_true in enumerate(d_list):
    lw_orig = original[d_true]
    latex_table += f"${d_true:4.1f}$ & ${lw_orig[0]:7.4f}$ & ${lw_orig[1]:7.4f}$ & ${lw_orig[2]:7.4f}$ & ${results[i, 0]:7.4f}$ & ${results[i, 1]:7.4f}$ & ${results[i, 2]:7.4f}$ \\\\\n"

latex_table += f"""\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\footnotesize
\\item Notes: Exact local Whittle estimator, $n = {n}$ observations from $\\ARFIMA(0,d,0)$, $m = \\lfloor n^{{{alpha:.2f}}} \\rfloor = {m}$ frequencies, {mc_reps:,} replications.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
"""

# Save LaTeX table
with open('tables/mc_elw.tex', 'w') as f:
    f.write(latex_table)

print("LaTeX table saved to: tables/mc_elw.tex")
