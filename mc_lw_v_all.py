#!/usr/bin/env python3
"""
Velasco (1999) tapered Local Whittle estimator Monte Carlo.
"""

import numpy as np

from pyelw import LW
from common import format_mse_latex, MSE_THRESHOLD
from pyelw.simulate import arfima

# Settings
n = 500
d_list = [-3.5, -2.3, -1.7, -1.3, -0.7, -0.3, 0.0, 0.3, 0.7, 1.3, 1.7, 2.3, 3.5]
mc_reps = 10000
alpha = 0.65
m = int(n**alpha)
bounds = (-4.0, 4.0)

# Estimators
lw = LW()

# Initialize storage for results
bartlett_est = np.zeros((mc_reps,))
cosine_est = np.zeros((mc_reps,))
kolmogorov_est = np.zeros((mc_reps,))
results_list = []  # For LaTeX generation

print("Replication of Table 2 of Shimotsu and Phillips (2005) (V Estimator)")
print(f"n={n}, m=n^{{0.65}}={m}, replications={mc_reps}, Bartlett taper")
print()
print("=====================================================================================")
print("|     |         Bartlett        |          Cosine         |        Kolmogorov       |")
print("|  d  |   bias    s.d.    MSE   |   bias    s.d.   MSE    |   bias    s.d.   MSE    |")
print("=====================================================================================")

# Loop over experiments and replications
for d_true in d_list:
    for rep in range(mc_reps):
        x = arfima(n, d_true, sigma=1.0, seed=42 + rep)
        result = lw.estimate(x, m=m, taper='bartlett', bounds=bounds, verbose=False)
        bartlett_est[rep] = result['d_hat']
        result = lw.estimate(x, m=m, taper='cosine', bounds=bounds, verbose=False)
        cosine_est[rep] = result['d_hat']
        result = lw.estimate(x, m=m, taper='kolmogorov', bounds=bounds, verbose=False)
        kolmogorov_est[rep] = result['d_hat']

    # Calculate results for each d value
    bartlett_bias = np.mean(bartlett_est) - d_true
    bartlett_sd = np.std(bartlett_est)
    bartlett_mse = np.mean((bartlett_est - d_true)**2)
    cosine_bias = np.mean(cosine_est) - d_true
    cosine_sd = np.std(cosine_est)
    cosine_mse = np.mean((cosine_est - d_true)**2)
    kolmogorov_bias = np.mean(kolmogorov_est) - d_true
    kolmogorov_sd = np.std(kolmogorov_est)
    kolmogorov_mse = np.mean((kolmogorov_est - d_true)**2)

    # Store for LaTeX generation
    results_list.append((d_true, bartlett_bias, bartlett_sd, bartlett_mse,
                        cosine_bias, cosine_sd, cosine_mse,
                        kolmogorov_bias, kolmogorov_sd, kolmogorov_mse))

    print(f"|{d_true:4.1f} "
          f"| {bartlett_bias:7.4f} {bartlett_sd:7.4f} {bartlett_mse:7.4f} "
          f"| {cosine_bias:7.4f} {cosine_sd:7.4f} {cosine_mse:7.4f} "
          f"| {kolmogorov_bias:7.4f} {kolmogorov_sd:7.4f} {kolmogorov_mse:7.4f} |")

print("=====================================================================================")

# Generate LaTeX table
latex_table = f"""\\begin{{table}}[htbp]
\\centering
\\begin{{threeparttable}}
\\caption{{Comparison of Velasco (1999) Tapers}}
\\label{{tab:mc:lw_v_all}}
\\begin{{tabular}}{{r@{{\\hspace{{1em}}}}rrr@{{\\hspace{{1em}}}}rrr@{{\\hspace{{1em}}}}rrr}}
\\toprule
& \\multicolumn{{3}}{{c}}{{Bartlett ($p=2$)}} & \\multicolumn{{3}}{{c}}{{Cosine ($p=3$)}} & \\multicolumn{{3}}{{c}}{{Kolmogorov ($p=3$)}} \\\\
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}} \\cmidrule(lr){{8-10}}
$d$ & Bias & S.D. & MSE & Bias & S.D. & MSE & Bias & S.D. & MSE \\\\
\\midrule
"""

for row in results_list:
    d_true, b_bias, b_sd, b_mse, c_bias, c_sd, c_mse, k_bias, k_sd, k_mse = row
    latex_table += f"${d_true:4.1f}$ & ${b_bias:7.4f}$ & ${b_sd:7.4f}$ & ${format_mse_latex(b_mse)}$ & ${c_bias:7.4f}$ & ${c_sd:7.4f}$ & ${format_mse_latex(c_mse)}$ & ${k_bias:7.4f}$ & ${k_sd:7.4f}$ & ${format_mse_latex(k_mse)}$ \\\\\n"

latex_table += f"""\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\footnotesize
\\item Notes: Comparison of three Velasco tapers, $n = {n}$ observations from $\\ARFIMA(0,d,0)$, $m = \\lfloor n^{{{alpha:.2f}}} \\rfloor = {m}$ frequencies, {mc_reps:,} replications. Shaded cells indicate $\\text{{MSE}} > {MSE_THRESHOLD:.2f}$.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
"""

# Save LaTeX table
with open('tables/mc_lw_v_all.tex', 'w') as f:
    f.write(latex_table)

print("LaTeX table saved to: tables/mc_lw_v_all.tex")
