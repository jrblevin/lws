#!/usr/bin/env python3
"""
Bandwidth Selection Analysis using S&P 500 Stock Index Prices

This script illustrates the Arteche and Orbe (2017) automatic bandwidth
selection procedure using the S&P 500 stock index series from
Hurvich and Chen (2000) with daily observations from July 1962
to December 1995.

Reference:

* Arteche, J. and J. Orbe (2017). A Strategy for Optimal Bandwidth Selection
  in Local Whittle Estimation. _Econometrics and Statistics_ 4, 3--17.

* Hurvich, C. M., and W. W. Chen (2000). An Efficient Taper for Potentially
  Overdifferenced Long-Memory Time Series. _Journal of Time Series Analysis_
  21, 155--180.
"""

import numpy as np
import matplotlib.pyplot as plt

from pyelw import LW, ELW, LWBootstrapM
from common import ESTIMATOR_COLORS

# Configuration
DATA_FILE = "data/snp500.dat"
OUTPUT_FIGURE = "figures/emp_bandwidth_selection.pdf"
OUTPUT_FIGURE_MSE = "figures/emp_bandwidth_mse.pdf"
OUTPUT_TABLE = "tables/emp_bandwidth_selection.tex"

# Bandwidth selection parameters
B = 200  # Bootstrap replications
M_MIN = 10
M_MAX = None  # Will be set to n/2

# Power rule bandwidth choices to compare
EXPONENTS = [0.5, 0.65, 0.8]

def main():
    # Load S&P 500 data
    series = np.loadtxt(DATA_FILE)
    n = len(series)
    print(f"S&P 500 Stock Prices")
    print(f"Sample size: n = {n}")
    print()

    # Set bandwidth range
    m_max = n // 2
    m_range = np.arange(M_MIN, m_max + 1)

    # Compute d_hat for each bandwidth using LW estimator
    print("Computing LW estimates across bandwidths...")
    lw = LW(bounds=(-0.5, 2.0))
    d_hat_lw = np.zeros(len(m_range))
    se_lw = np.zeros(len(m_range))

    for i, m in enumerate(m_range):
        result = lw.estimate(series, m=m, verbose=False)
        d_hat_lw[i] = result['d_hat']
        se_lw[i] = result['se']
        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{len(m_range)} bandwidths")

    # Run bootstrap bandwidth selection
    print()
    print("Running Arteche-Orbe bootstrap bandwidth selection...")
    print(f"Bootstrap replications: B = {B}")

    selector = LWBootstrapM(
        lw_estimator=lw,
        B=B,
        m_min=M_MIN,
        m_max=m_max,
        verbose=True
    )
    selector.fit(series)

    m_star = selector.optimal_m_
    d_star = selector.d_hat_
    se_star = selector.se_
    k_n = selector.k_n_
    iterations = selector.iterations_
    converged = selector.converged_

    print()
    print("Bootstrap Bandwidth Selection Results")
    print("=====================================")
    print()
    print(f"Optimal bandwidth: m* = {m_star}")
    print(f"Memory parameter:  d* = {d_star:.4f} (SE = {se_star:.4f})")
    print(f"Resampling width:  k_n = {k_n}")
    print(f"Iterations:        {iterations}")
    print(f"Converged:         {converged}")
    print()

    # Compute power rule bandwidth choices
    print("Power Rule Bandwidths")
    print("---------------------")
    print()
    power_rule_results = []
    for alpha in EXPONENTS:
        m_power = round(n ** alpha)
        result = lw.estimate(series, m=m_power, verbose=False)
        power_rule_results.append({
            'alpha': alpha,
            'm': m_power,
            'd_hat': result['d_hat'],
            'se': result['se'],
            'ci_lower': result['d_hat'] - 1.96 * result['se'],
            'ci_upper': result['d_hat'] + 1.96 * result['se'],
        })
        # Note: alpha = 0.8 matches Hurvich and Chen (2000)
        hc_note = " (HC 2000)" if alpha == 0.8 else ""
        print(f"m = n^{alpha:.2f} = {m_power:4d}:  d = {result['d_hat']:.4f} (SE = {result['se']:.4f}){hc_note}")

    #
    # Bandwidth selection figure
    #
    print()
    print("Generating bandwidth selection figure...")

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot d_hat vs m (using LW color from common scheme)
    ax.plot(m_range, d_hat_lw, color=ESTIMATOR_COLORS['LW'], linewidth=1.5,
            alpha=0.8, label=r'$\hat{d}_{\text{LW}}(m)$')

    # Add confidence band (approximate 95% CI)
    upper = d_hat_lw + 1.96 * se_lw
    lower = d_hat_lw - 1.96 * se_lw
    ax.fill_between(m_range, lower, upper, alpha=0.15,
                    color=ESTIMATOR_COLORS['LW'], label='95% CI')

    # Vertical lines at power rule choices (using other estimator colors)
    power_rule_colors = [
        ESTIMATOR_COLORS['V'],      # Orange for n^0.5
        ESTIMATOR_COLORS['HC'],     # Green for n^0.65
        ESTIMATOR_COLORS['2ELW'],   # Purple for n^0.8 (HC 2000)
    ]
    linestyles = [':', '-.', '--']
    for i, res in enumerate(power_rule_results):
        ax.axvline(x=res['m'], color=power_rule_colors[i], linestyle=linestyles[i],
                   linewidth=1.5, alpha=0.8,
                   label=f"$m = n^{{{res['alpha']:.2f}}} = {res['m']}$")

    # Vertical line at optimal m*
    ax.axvline(x=m_star, color=ESTIMATOR_COLORS['ELW'], linestyle='-',
               linewidth=1.5, alpha=0.8,
               label=f'$m^* = {m_star}$ (Bootstrap)')

    # Horizontal line at d = 1 (unit root)
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(m_range[-1] * 0.98, 1.02, '$d = 1$', ha='right', va='bottom',
            fontsize=9, color='gray')

    # Labels and formatting
    ax.set_xlabel('Bandwidth $m$', fontsize=11)
    ax.set_ylabel(r'$\hat{d}_{\text{LW}}$', fontsize=11)
    # ax.set_title('S&P 500: Local Whittle Estimates by Bandwidth', fontsize=12)

    # Set axis limits
    ax.set_xlim(0, m_max)
    ax.set_ylim(0.7, 1.3)

    # Legend
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches='tight')
    print(f"Bandwidth selection figure saved to: {OUTPUT_FIGURE}")

    # Generate LaTeX table
    latex_table = generate_latex_table(m_star, d_star, se_star, power_rule_results, n, k_n)
    with open(OUTPUT_TABLE, 'w') as f:
        f.write(latex_table)
    print(f"Bandwidth selection table saved to: {OUTPUT_TABLE}")

    #
    # MSE profile figure
    #
    print("Generating MSE profile figure...")

    # Extract MSE profile from selector
    mse_profile = selector.mse_profile_
    mse_bandwidths = np.array(sorted(mse_profile.keys()))
    mse_values = np.array([mse_profile[m] for m in mse_bandwidths])

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    # Plot MSE vs m
    ax2.plot(mse_bandwidths, mse_values, color=ESTIMATOR_COLORS['LW'], linewidth=1.5,
             alpha=0.8, label='Bootstrap MSE')

    # Vertical line at optimal m*
    ax2.axvline(x=m_star, color=ESTIMATOR_COLORS['ELW'], linestyle='-', linewidth=1.5,
                alpha=0.8, label=f'$m^* = {m_star}$')

    # Horizontal line at minimum MSE
    min_mse = mse_profile[m_star]
    ax2.axhline(y=min_mse, color=ESTIMATOR_COLORS['ELW'], linestyle=':', linewidth=1,
                alpha=0.5)

    # Vertical lines at power rule choices
    for i, res in enumerate(power_rule_results):
        ax2.axvline(x=res['m'], color=power_rule_colors[i], linestyle=linestyles[i],
                    linewidth=1.5, alpha=0.8,
                    label=f"$m = n^{{{res['alpha']:.2f}}} = {res['m']}$")

    # Labels and formatting
    ax2.set_xlabel('Bandwidth $m$', fontsize=11)
    ax2.set_ylabel('Bootstrap MSE Profile', fontsize=11)

    # Set axis limits
    ax2.set_xlim(0, mse_bandwidths[-1])

    # Legend
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # Grid
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE_MSE, dpi=300, bbox_inches='tight')
    print(f"MSE profile figure saved to: {OUTPUT_FIGURE_MSE}")


def generate_latex_table(m_star, d_star, se_star, power_rule_results, n, k_n):
    latex = r"""\begin{table}[t!]
\centering
\begin{threeparttable}
\caption{S\&P 500: Bandwidth Selection Comparison}
\label{tab:emp_bandwidth_selection}
\begin{tabular}{llrrrr}
\toprule
Method & Rule & $m$ & $\hat{d}_{\text{LW}}$ & SE & 95\% CI \\
\midrule
"""
    # Power rules n^\alpha. Hurvich and Chen (2000) used alpha = 0.8.
    for i, res in enumerate(power_rule_results):
        label = "Power Rule" if i == 0 else ""
        # hc_note = " (HC 2000)" if res['alpha'] == 0.8 else ""
        hc_note = ""
        latex += f"{label} & $m = n^{{{res['alpha']:.2f}}}${hc_note} & {res['m']} & ${res['d_hat']:.3f}$ & ${res['se']:.3f}$ & $({res['ci_lower']:.3f},{res['ci_upper']:.3f})$ \\\\\n"

    # Arteche and Orbe (2017) bandwidth
    ci_lower_star = d_star - 1.96 * se_star
    ci_upper_star = d_star + 1.96 * se_star
    latex += r"\midrule" + "\n"
    latex += f"\\cite{{arteche-orbe-2017}} & Min. Bootstrap MSE & {m_star} & ${d_star:.3f}$ & ${se_star:.3f}$ & $({ci_lower_star:.3f},{ci_upper_star:.3f})$ \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item Notes: Logarithm of S\&P 500 daily stock index value, $n = """ + str(n) + r"""$. \cite{hurvich-chen-2000}
used $m=n^{0.80}$. Bootstrap MSE bandwidth
selected by minimizing bootstrap MSE with $B = """ + str(B) + r"""$ replications and resampling
width $k_n = """ + str(k_n) + r"""$ following \cite{arteche-orbe-2017}.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    return latex


if __name__ == "__main__":
    main()
