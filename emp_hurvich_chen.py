#!/usr/bin/env python3
"""
Comprehensive Estimator Comparison on Hurvich-Chen (2000) Datasets

This script replicates the results of Table III in Hurvich and Chen (2000)
by comparing all Local Whittle estimators across the same datasets.
Hurvich and Chen (2000) originally reported results for the HC tapered LW
estimator and the LW estimator on differenced data.

Reference:

Hurvich, C. M., and W. W. Chen (2000). An Efficient Taper for Potentially
Overdifferenced Long-Memory Time Series. _Journal of Time Series Analysis_,
21, 155--180.
"""

import numpy as np

from pyelw import LW, ELW, TwoStepELW

# Dataset specifications from Hurvich-Chen (2000) Table III
specifications = [
    # Description,     filename,          transform,     m, d_hat, se
    ('Global temp.',   'glotemp.dat',     None,        130,  0.45, 0.060),
    ('S&P 500',        'snp500.dat',      None,       1383,  0.99, 0.018),
    ('Inflation, US',  'cpi_us.dat',      'diff-log',   40,  0.57, 0.123),
    ('Inflation, UK',  'cpi_uk.dat',      'diff-log',   40,  0.33, 0.123),
    ('Inflation, FR',  'cpi_fr.dat',      'diff-log',   40,  0.67, 0.123),
    ('Real wages, US', 'realwage_us.dat', None,         35,  1.43, 0.121),
    ('Ind. prod., US', 'indpro_us.dat',   'log',       100,  1.34, 0.075),
]

bounds = (-2.0, 2.0)

# Store results for LaTeX generation
latex_results = []

print("Hurvich-Chen (2000) Datasets: Estimator Comparison")
print("==================================================")
print()
print("|                 |      |      |  HC (2000)   |                                  PyELW                                   |")
print(f"| {'Series':<15} | {'n':>4} | {'m':>4} | {'HC':>12} | {'LW':>12} | {'V':>12} | {'HC':>12} | {'ELW':>12} | {'2ELW':>12} |")
print("|-----------------|------|------|--------------|--------------|--------------|--------------|--------------|--------------|")

for desc, filename, transform, m, paper_d_hat, paper_se in specifications:
    # Load data
    series = np.loadtxt(f"data/{filename}")
    n = len(series) - 1 if transform == 'diff-log' else len(series)

    # Apply transformation
    if transform == 'log':
        series = np.log(series)
    elif transform == 'diff-log':
        series = np.diff(np.log(series))

    # Estimate with all methods
    # LW estimator (no taper)
    lw = LW(taper='none')
    lw_result = lw.estimate(series, m=m, bounds=bounds, verbose=False)
    lw_d_hat = lw_result['d_hat']
    lw_se = lw_result['se']

    # Note: Hurvich and Chen (2000) report LW estimates with first-differenced data,
    # then add 1 back to the estimate.  To replicate those results, use the following
    # instead:
    # diff_series = np.diff(series)
    # lw_result = lw.estimate(diff_series, m=m, bounds=bounds, verbose=False)
    # lw_d_hat = lw_result['d_hat'] + 1
    # lw_se = lw_result['se']

    # Velasco (Kolmogorov taper)
    lw_v = LW(taper='kolmogorov')
    v_result = lw_v.estimate(series, m=m, bounds=bounds, verbose=False)
    v_d_hat = v_result['d_hat']
    v_se = v_result['se']

    # Hurvich-Chen taper
    lw_hc = LW(taper='hc')
    hc_result = lw_hc.estimate(series, m=m, bounds=bounds, verbose=False)
    hc_d_hat = hc_result['d_hat']
    hc_se = hc_result['se']

    # Exact Local Whittle
    elw = ELW()
    elw_result = elw.estimate(series, m=m, bounds=bounds, verbose=False)
    elw_d_hat = elw_result['d_hat']
    elw_se = elw_result['se']

    # Two-step Exact Local Whittle
    elw2 = TwoStepELW()
    elw2_result = elw2.estimate(series, m=m, bounds=bounds, trend_order=0, verbose=False)
    elw2_d_hat = elw2_result['d_hat']
    elw2_se = elw2_result['se']

    # Store for LaTeX generation
    latex_results.append({
        'desc': desc,
        'n': n,
        'm': m,
        'paper_d': paper_d_hat,
        'paper_se': paper_se,
        'lw_d': lw_d_hat,
        'lw_se': lw_se,
        'v_d': v_d_hat,
        'v_se': v_se,
        'hc_d': hc_d_hat,
        'hc_se': hc_se,
        'elw_d': elw_d_hat,
        'elw_se': elw_se,
        'elw2_d': elw2_d_hat,
        'elw2_se': elw2_se
    })

    # Print results
    print(f"| {desc:<15} | {n:>4d} | {m:>4d} | {paper_d_hat:>12.2f} | {lw_d_hat:>12.2f} | {v_d_hat:>12.2f} | {hc_d_hat:>12.2f} | {elw_d_hat:>12.2f} | {elw2_d_hat:>12.2f} |")
    print(f"| {'':>15} | {'':>4} | {'':>4} |      ({paper_se:>5.3f}) |      ({lw_se:>5.3f}) |      ({v_se:>5.3f}) |      ({hc_se:>5.3f}) |      ({elw_se:>5.3f}) |      ({elw2_se:>5.3f}) |")

# Generate LaTeX table
latex_table = """\\begin{table}[t!]
\\centering
\\begin{threeparttable}
\\caption{Hurvich and Chen (2000) Datasets: LW Estimator Comparison}
\\label{tab:emp_hurvich_chen}
\\scriptsize
\\begin{tabular}{lrr|c|ccccc}
\\toprule
& & & Original & \\multicolumn{5}{c}{Replication} \\\\
\\cmidrule(lr){4-4} \\cmidrule(lr){5-9}
Series & $n$ & $m$ & HC & LW & V & HC & ELW & 2ELW \\\\
\\midrule
"""

for res in latex_results:
    # Escape LaTeX special characters in description
    desc_escaped = res['desc'].replace('&', '\\&')
    latex_table += f"{desc_escaped} & {res['n']} & {res['m']} & ${res['paper_d']:0.2f}$ & ${res['lw_d']:0.2f}$ & ${res['v_d']:0.2f}$ & ${res['hc_d']:0.2f}$ & ${res['elw_d']:0.2f}$ & ${res['elw2_d']:0.2f}$ \\\\\n"
    latex_table += f"& & & $({res['paper_se']:0.3f})$ & $({res['lw_se']:0.3f})$ & $({res['v_se']:0.3f})$ & $({res['hc_se']:0.3f})$ & $({res['elw_se']:0.3f})$ & $({res['elw2_se']:0.3f})$ \\\\\n"

latex_table += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\footnotesize
\\item Notes: Estimates of $d$ with standard errors in parentheses. Original column shows results from Hurvich and Chen (2000) Table III for HC. Replication columns report results from multiple estimators: LW = Local Whittle, V = Velasco (Kolmogorov), HC = Hurvich-Chen, ELW = Exact Local Whittle, 2ELW = Two-step ELW.
\\end{tablenotes}
\\end{threeparttable}
\\end{table}
"""

# Save LaTeX table
with open('tables/emp_hurvich_chen.tex', 'w') as f:
    f.write(latex_table)

print("LaTeX table saved to: tables/emp_hurvich_chen.tex")
