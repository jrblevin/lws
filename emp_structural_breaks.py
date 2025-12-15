#!/usr/bin/env python3
"""
Structural break analysis using French inflation data

This script analyzes the CPI data for France from Hurvich and Chen (2000)
to demonstrate how structural breaks can cause method disagreement and
spurious long memory.  It applies Bai-Perron (1998, 2003) break detection
and compares full-sample vs subsample estimates across all Local Whittle
estimators.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta

from pyelw import LW, ELW, TwoStepELW, LWBootstrapM
from common import ESTIMATOR_COLORS
from bai_perron import find_breakpoints, select_breaks_bic
from qu_test import qu_test

# Configuration
DATA_FILE = "data/cpi_fr.dat"
OUTPUT_FIGURE = "figures/emp_structural_breaks.pdf"
OUTPUT_TABLE = "tables/emp_structural_breaks.tex"

# Estimation parameters
BOUNDS = (-2.0, 2.0)
M_FULL = 40  # Bandwidth for full sample from Hurvich and Chen (2000)

# Break detection parameters
MAX_BREAKS = 7  # Maximum number of breaks to consider
MIN_SEGMENT_SIZE = 24  # Minimum segment size (2 years of monthly data)

# Start date of series (January 1957)
START_DATE = datetime(1957, 1, 1)


def index_to_date(index):
    """Convert array index to date string."""
    date = START_DATE + relativedelta(months=index)
    return date.strftime("%Y-%m")


def estimate_all_methods(series, m):
    """Estimate d using all five Local Whittle methods."""
    results = {}

    # LW estimator (no taper)
    lw = LW(taper='none')
    lw_result = lw.estimate(series, m=m, bounds=BOUNDS, verbose=False)
    results['LW'] = {'d': lw_result['d_hat'], 'se': lw_result['se']}

    # Velasco (Kolmogorov taper)
    lw_v = LW(taper='kolmogorov')
    v_result = lw_v.estimate(series, m=m, bounds=BOUNDS, verbose=False)
    results['V'] = {'d': v_result['d_hat'], 'se': v_result['se']}

    # Hurvich-Chen taper
    lw_hc = LW(taper='hc')
    hc_result = lw_hc.estimate(series, m=m, bounds=BOUNDS, verbose=False)
    results['HC'] = {'d': hc_result['d_hat'], 'se': hc_result['se']}

    # Exact Local Whittle
    elw = ELW()
    elw_result = elw.estimate(series, m=m, bounds=BOUNDS, verbose=False)
    results['ELW'] = {'d': elw_result['d_hat'], 'se': elw_result['se']}

    # Two-step Exact Local Whittle
    elw2 = TwoStepELW()
    elw2_result = elw2.estimate(series, m=m, bounds=BOUNDS, trend_order=0, verbose=False)
    results['2ELW'] = {'d': elw2_result['d_hat'], 'se': elw2_result['se']}

    return results




def main():
    # Load French CPI data and first difference logs
    cpi = np.loadtxt(DATA_FILE)
    inflation = np.diff(np.log(cpi))
    n = len(inflation)

    print("Inflation in France: Structural Break Analysis")
    print("==============================================")
    print()
    print(f"Series: Monthly diff(log(CPI))")
    print(f"Period: {index_to_date(0)} to {index_to_date(n-1)}")
    print(f"Sample size: n = {n}")
    print()

    #
    # Part 1: Break Detection using Bai-Perron (1998, 2003)
    #
    print("Part 1: Bai-Perron Break Detection")
    print("----------------------------------")
    print()

    # Use Bai-Perron dynamic programming for globally optimal breaks
    trim = MIN_SEGMENT_SIZE / n  # Convert to fraction

    # Find optimal number of breaks using BIC
    print("BIC-based break selection (Bai-Perron dynamic programming):")
    print()
    optimal_k, bic_results = select_breaks_bic(inflation, exog=None,
                                                max_breaks=MAX_BREAKS, trim=trim)
    for result in bic_results:
        print(f"- k = {result['nbreaks']}: BIC = {result['bic']:.2f}")

    print()
    print(f"Optimal number of breaks (BIC): k = {optimal_k}")
    print()

    # Get break locations for k=2 (matches economic events)
    k_analysis = 2
    breakpoints, ssr = find_breakpoints(inflation, exog=None,
                                         nbreaks=k_analysis, trim=trim)
    # Convert from 0-indexed last-obs-in-regime to 0-indexed first-obs-in-new-regime
    # breakpoints[i] is the last observation in regime i
    # So regime i+1 starts at breakpoints[i] + 1
    breakpoints_list = list(breakpoints)

    print(f"Using k = {k_analysis} breaks for analysis:")
    print()
    for i, bp in enumerate(breakpoints_list):
        # bp is the last observation in regime i, so regime i+1 starts at bp+1
        print(f"- Break {i+1}: {index_to_date(bp+1)} (regime {i+2} starts at index {bp+1})")
    print()

    # Use bp+1 as the start of new regime for defining segments
    breakpoints = [bp + 1 for bp in breakpoints_list]

    #
    # Part 2: Regime Characteristics
    #
    print("Part 2: Regime Characteristics")
    print("------------------------------")
    print()

    # Define regimes (breakpoints now contains first obs of new regime)
    regimes = [
        {'name': 'Pre-oil shocks', 'start': 0, 'end': breakpoints[0],
         'period': f"1957-01 to {index_to_date(breakpoints[0]-1)}"},
        {'name': 'Oil shocks', 'start': breakpoints[0], 'end': breakpoints[1],
         'period': f"{index_to_date(breakpoints[0])} to {index_to_date(breakpoints[1]-1)}"},
        {'name': 'Disinflation', 'start': breakpoints[1], 'end': n,
         'period': f"{index_to_date(breakpoints[1])} to {index_to_date(n-1)}"},
    ]

    print(f"{'Regime':<20} {'Period':<25} {'N':>5} {'Mean (ann.)':>12} {'Std (ann.)':>12}")
    print("------------------------------------------------------------------------------")

    regime_stats = []
    for regime in regimes:
        segment = inflation[regime['start']:regime['end']]
        mean_ann = np.mean(segment) * 12 * 100  # Annualized percentage
        std_ann = np.std(segment) * np.sqrt(12) * 100  # Annualized percentage
        regime['n'] = len(segment)
        regime['mean_ann'] = mean_ann
        regime['std_ann'] = std_ann
        regime_stats.append(regime)
        print(f"{regime['name']:<20} {regime['period']:<25} {regime['n']:>5} {mean_ann:>11.1f}% {std_ann:>11.1f}%")

    print()

    #
    # Part 3: Full-sample and Subsample Estimation
    #
    print("Part 3: Local Whittle Estimates")
    print("-------------------------------")
    print()

    # Full sample estimates
    print("Full sample estimates (n = {}, m = {}):".format(n, M_FULL))
    full_results = estimate_all_methods(inflation, M_FULL)
    print()
    print(f"{'Method':<8} {'d_hat':>8} {'SE':>8}")
    print("--------------------------")
    for method in ['LW', 'V', 'HC', 'ELW', '2ELW']:
        r = full_results[method]
        print(f"{method:<8} {r['d']:>8.3f} {r['se']:>8.3f}")
    print()

    # Compute alpha from full sample: m = n^alpha
    alpha = np.log(M_FULL) / np.log(n)
    print(f"Implied alpha from full sample: {alpha:.4f}")
    print()

    # Panel A: Subsample estimates with same alpha (m_sub = n_sub^alpha)
    print("Panel A: Subsample estimates (same alpha)")
    print("-----------------------------------------")
    print()

    subsample_results_alpha = []
    for regime in regime_stats:
        segment = inflation[regime['start']:regime['end']]
        n_sub = len(segment)

        # Same alpha: m_sub = round(n_sub^alpha)
        m_sub = max(10, round(n_sub ** alpha))

        results = estimate_all_methods(segment, m_sub)
        results['regime'] = regime['name']
        results['n'] = n_sub
        results['m'] = m_sub
        results['period'] = regime['period']
        subsample_results_alpha.append(results)

        print(f"{regime['name']} (n = {n_sub}, m = {m_sub}):")
        print()
        print(f"{'Method':<8} {'d_hat':>8} {'SE':>8}")
        print("--------------------------")
        for method in ['LW', 'V', 'HC', 'ELW', '2ELW']:
            r = results[method]
            print(f"{method:<8} {r['d']:>8.3f} {r['se']:>8.3f}")
        print()

    # Panel B: Subsample estimates with bootstrap-optimal bandwidth
    print("Panel B: Subsample estimates (bootstrap-MSE-minimum m)")
    print("------------------------------------------------------")
    print()

    subsample_results_bootstrap = []
    for regime in regime_stats:
        segment = inflation[regime['start']:regime['end']]
        n_sub = len(segment)

        # Bootstrap-optimal bandwidth using LWBootstrapM
        print(f"{regime['name']}: Computing bootstrap-optimal bandwidth...")
        bootstrap_lw = LWBootstrapM(bounds=BOUNDS, B=200, verbose=False)
        bootstrap_lw.fit(segment)
        m_opt = bootstrap_lw.optimal_m_
        print(f"  Optimal m = {m_opt}")

        results = estimate_all_methods(segment, m_opt)
        results['regime'] = regime['name']
        results['n'] = n_sub
        results['m'] = m_opt
        results['period'] = regime['period']
        subsample_results_bootstrap.append(results)

        print()
        print(f"{'Method':<8} {'d_hat':>8} {'SE':>8}")
        print("--------------------------")
        for method in ['LW', 'V', 'HC', 'ELW', '2ELW']:
            r = results[method]
            print(f"{method:<8} {r['d']:>8.3f} {r['se']:>8.3f}")
        print()

    # Use alpha-based results for figure (consistent with full sample)
    subsample_results = subsample_results_alpha

    #
    # Part 4: Qu (2011) Test for Spurious Long Memory
    #
    print("Part 4: Qu (2011) Test for Spurious Long Memory")
    print("-----------------------------------------------")
    print()
    print("Null hypothesis: True long memory (fractional integration)")
    print("Alternative: Spurious long memory (e.g., structural breaks)")
    print()

    # Test full sample
    m_qu = int(np.floor(1 + n**0.75))  # Standard bandwidth for Qu test
    qu_full = qu_test(inflation, m=m_qu, epsilon=0.05)
    print(f"Full sample (n = {n}, m = {m_qu}):")
    print(f"  d_hat = {qu_full['d_hat']:.4f}")
    print(f"  W statistic = {qu_full['W_stat']:.4f}")
    print(f"  Critical values (eps=0.05): 10%: {qu_full['critical_values']['10%']}, "
          f"5%: {qu_full['critical_values']['5%']}, 1%: {qu_full['critical_values']['1%']}")
    print(f"  Reject at 5%: {qu_full['reject_05']}")
    print()

    # Test subsamples
    print("Subsample tests:")
    print()
    qu_subsample_results = []
    for regime in regime_stats:
        segment = inflation[regime['start']:regime['end']]
        n_sub = len(segment)
        m_qu_sub = int(np.floor(1 + n_sub**0.75))

        qu_result = qu_test(segment, m=m_qu_sub, epsilon=0.05)
        qu_subsample_results.append({
            'regime': regime['name'],
            'n': n_sub,
            'm': m_qu_sub,
            'W_stat': qu_result['W_stat'],
            'd_hat': qu_result['d_hat'],
            'reject_05': qu_result['reject_05'],
            'reject_10': qu_result['reject_10']
        })

        print(f"{regime['name']} (n = {n_sub}, m = {m_qu_sub}):")
        print(f"  d_hat = {qu_result['d_hat']:.4f}")
        print(f"  W statistic = {qu_result['W_stat']:.4f}")
        print(f"  Reject at 10%: {qu_result['reject_10']}, at 5%: {qu_result['reject_05']}")
        print()

    #
    # Part 5: Generate Figure
    #
    print("Part 5: Generating figure...")

    fig, ax = plt.subplots(figsize=(10, 5))

    # Create time axis (months from start)
    time_axis = np.arange(n)

    # Plot inflation series
    ax.plot(time_axis, inflation * 12 * 100, color=ESTIMATOR_COLORS['LW'],
            linewidth=0.8, alpha=1.0, label='Monthly inflation (ann.)')

    # Plot regime means
    colors = [ESTIMATOR_COLORS['V'], ESTIMATOR_COLORS['ELW'], ESTIMATOR_COLORS['2ELW']]
    for i, regime in enumerate(regime_stats):
        start, end = regime['start'], regime['end']
        mean_val = regime['mean_ann']
        ax.hlines(mean_val, start, end - 1, colors=colors[i], linewidth=2.5, alpha=0.7,
                  label=f"{regime['name']}: {mean_val:.1f}%")

    # Add vertical lines at break points
    for bp in breakpoints:
        ax.axvline(x=bp, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # Add break date labels
    for i, bp in enumerate(breakpoints):
        date_str = index_to_date(bp)
        ax.text(bp + 2, ax.get_ylim()[1] * 0.95, date_str,
                fontsize=9, rotation=0, va='top')

    # Format x-axis with years
    year_ticks = []
    year_labels = []
    for year in range(1960, 2000, 5):
        month_idx = (year - 1957) * 12
        if 0 <= month_idx < n:
            year_ticks.append(month_idx)
            year_labels.append(str(year))
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(year_labels)

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Inflation Rate (% per year)', fontsize=11)
    ax.set_xlim(0, n)

    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {OUTPUT_FIGURE}")

    #
    # Part 6: Generate LaTeX Table
    #
    print("Generating LaTeX table...")

    latex = generate_latex_table(n, M_FULL, alpha, full_results,
                                  subsample_results_alpha, subsample_results_bootstrap,
                                  breakpoints)
    with open(OUTPUT_TABLE, 'w') as f:
        f.write(latex)
    print(f"Table saved to: {OUTPUT_TABLE}")


def generate_latex_table(n, m, alpha, full_results, subsample_results_alpha,
                         subsample_results_bootstrap, breakpoints):
    """Generate LaTeX table for the paper with two panels."""

    latex = r"""\begin{table}[t!]
\centering
\begin{threeparttable}
\caption{Inflation in France: Subsample Analysis with Detected Structural Breaks}
\label{tab:french_subsample}
\begin{tabular}{lrrccccc}
\toprule
Period & $n$ & $m$ & LW & V & HC & ELW & 2ELW \\
\midrule
"""

    # Full sample row
    latex += f"Full sample (1957--1997) & {n} & {m}"
    for method in ['LW', 'V', 'HC', 'ELW', '2ELW']:
        latex += f" & ${full_results[method]['d']:.3f}$"
    latex += r" \\" + "\n"

    # Standard errors for full sample
    latex += f" & & "
    for method in ['LW', 'V', 'HC', 'ELW', '2ELW']:
        latex += f" & $({full_results[method]['se']:.3f})$"
    latex += r" \\" + "\n"

    # Panel A header
    latex += r"\midrule" + "\n"
    latex += r"\multicolumn{8}{c}{\textit{Panel A: Power rule bandwidth ($m = n^{" + f"{alpha:.2f}" + r"}$)}} \\" + "\n"
    latex += r"\midrule" + "\n"

    # Panel A: Subsample rows with same alpha
    for res in subsample_results_alpha:
        latex += f"{res['regime']} & {res['n']} & {res['m']}"
        for method in ['LW', 'V', 'HC', 'ELW', '2ELW']:
            d_val = res[method]['d']
            if d_val < 0:
                latex += f" & $-{abs(d_val):.3f}$"
            else:
                latex += f" & ${d_val:.3f}$"
        latex += r" \\" + "\n"

        # Standard errors
        latex += f" & & "
        for method in ['LW', 'V', 'HC', 'ELW', '2ELW']:
            latex += f" & $({res[method]['se']:.3f})$"
        latex += r" \\" + "\n"

    # Panel B header
    latex += r"\midrule" + "\n"
    latex += r"\multicolumn{8}{c}{\textit{Panel B: Bootstrap-MSE-optimal bandwidth $m^*$}} \\" + "\n"
    latex += r"\midrule" + "\n"

    # Panel B: Subsample rows with bootstrap-optimal bandwidth
    for res in subsample_results_bootstrap:
        latex += f"{res['regime']} & {res['n']} & {res['m']}"
        for method in ['LW', 'V', 'HC', 'ELW', '2ELW']:
            d_val = res[method]['d']
            if d_val < 0:
                latex += f" & $-{abs(d_val):.3f}$"
            else:
                latex += f" & ${d_val:.3f}$"
        latex += r" \\" + "\n"

        # Standard errors
        latex += f" & & "
        for method in ['LW', 'V', 'HC', 'ELW', '2ELW']:
            latex += f" & $({res[method]['se']:.3f})$"
        latex += r" \\" + "\n"

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item Notes: Estimates of $d$ with standard errors in parentheses. French
inflation is monthly diff-log CPI from January 1957 to December 1997.
Structural breaks detected following \cite{bai-perron-2003}: """

    # Add break dates
    break_dates = [index_to_date(bp) for bp in breakpoints]
    latex += ", ".join(break_dates)

    latex += r""". Mean inflation: 5.0\%/yr (pre-1973), 10.3\%/yr (1973--1984), 2.5\%/yr (post-1984).
Panel A uses the same power rule as the full sample.
Panel B uses bootstrap MSE-optimal bandwidth selection \citep{arteche-orbe-2017}.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""

    return latex


if __name__ == "__main__":
    main()
