#!/usr/bin/env python3
"""
Local Whittle estimator Monte Carlo

This is a replication of LW Monte Carlo in the right panel of Table 1
of Shimotsu and Phillips (2005).
"""

import numpy as np

from pyelw import LW
from pyelw.simulate import arfima

# Settings
n = 500
d_list = [-1.3, -0.7, -0.3, 0.0, 0.3, 0.7, 1.3]
mc_reps = 10000
alpha = 0.65
m = int(n**alpha)

# Estimators
lw = LW()

# Initialize storage for results
lw_estimates = np.zeros((mc_reps, len(d_list)))

# Results matrices: (bias, se, mse) for each d value
lw_results = np.zeros((len(d_list), 3))

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

print("Replication of Table 1 of Shimotsu and Phillips (2005) (Right Panel)")
print(f"n={n}, m=n^{{0.65}}={m}, replications={mc_reps}")
print()
print("============================================================")
print("|     |        SP (2005)         |          PyELW          |")
print("|  d  |   bias    s.d.    MSE    |   bias    s.d.   MSE    |")
print("============================================================")

# Loop over experiments
for i, d_true in enumerate(d_list):
    lw_orig = original_lw[d_true]

    for rep in range(mc_reps):
        x = arfima(n, d_true, sigma=1.0, seed=42 * i + rep)
        lw_result = lw.estimate(x, m=m, bounds=(-4.0, 4.0), verbose=False)
        lw_estimates[rep, i] = lw_result['d_hat']

    # Calculate results for each d value
    lw_results[i, 0] = np.mean(lw_estimates[:, i]) - d_true  # Bias
    lw_results[i, 1] = np.std(lw_estimates[:, i])  # S.E.
    lw_results[i, 2] = np.mean((lw_estimates[:, i] - d_true)**2)  # MSE

    print(f"|{d_true:4.1f} |  {lw_orig[0]:7.4f} {lw_orig[1]:7.4f} {lw_orig[2]:7.4f} "
          f"| {lw_results[i, 0]:7.4f} {lw_results[i, 1]:7.4f} {lw_results[i, 2]:7.4f} |")

print("============================================================")

# Generate LaTeX table
latex_table = f"""\\begin{{table}}[t!]
\\centering
\\begin{{threeparttable}}
\\caption{{LW Estimator: Replication of Right Panel of Table 1 of Shimotsu and Phillips (2005)}}
\\label{{tab:mc:lw}}
\\begin{{tabular}}{{r@{{\\hspace{{1em}}}}rrr@{{\\hspace{{1em}}}}rrr}}
\\toprule
\\multicolumn{{1}}{{c}}{{}} & \\multicolumn{{3}}{{c}}{{Original}} & \\multicolumn{{3}}{{c}}{{Replication}} \\\\
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}}
$d$ & Bias & S.D. & MSE & Bias & S.D. & MSE \\\\
\\midrule
"""

for i, d_true in enumerate(d_list):
    lw_orig = original_lw[d_true]
    latex_table += f"${d_true:4.1f}$ & ${lw_orig[0]:7.4f}$ & ${lw_orig[1]:7.4f}$ & ${lw_orig[2]:7.4f}$ & ${lw_results[i, 0]:7.4f}$ & ${lw_results[i, 1]:7.4f}$ & ${lw_results[i, 2]:7.4f}$ \\\\\n"

latex_table += f"""\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\footnotesize
\\item Notes: LW estimator, $n = {n}$ observations from $\\ARFIMA(0,d,0)$, $m = \\lfloor n^{{{alpha:.2f}}} \\rfloor = {m}$ frequencies, {mc_reps:,} replications.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
"""

# Save LaTeX table
with open('tables/mc_lw.tex', 'w') as f:
    f.write(latex_table)

print("LaTeX table saved to: tables/mc_lw.tex")
