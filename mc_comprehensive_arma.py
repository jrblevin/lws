#!/usr/bin/env python3
"""
Estimator Comparison under ARMA(1,1) Short-Run Dynamics

Monte Carlo comparison of Local Whittle estimators with ARFIMA(1,d,1) data.
Companion to mc_comprehensive.py (AR(1)) and mc_comprehensive_ma.py (MA(1)),
used for the short-run-specification robustness check. The design, estimators,
parameter grid, bandwidth, and seed scheme match mc_comprehensive.py, but the
single AR(1) coefficient is replaced by a short list of (phi, theta)
configurations, the last of which is a difficult near-unit-root case.
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool

from pyelw import LW, ELW, TwoStepELW, LWLFC
from common import format_mse_latex, MSE_THRESHOLD, arfima_arma

# Settings (short-run config is now (phi, theta))
n_obs = 500
d_list = [-2.2, -1.8, -1.2, -0.6, -0.3, 0.0, 0.3, 0.6, 1.2, 1.8, 2.2]
# (phi, theta) pairs, increasing difficulty
arma_configs = [(0.5, 0.5), (0.8, 0.5), (0.9, 0.5)]
mc_reps = 10000
alpha = 0.65
m = int(n_obs**alpha)
bounds = (-4.0, 4.0)
seed_base = 42

estimator_names = ['LW', 'V', 'HC', 'ELW', '2ELW', 'LWLFC']


def run_single_rep(args):
    """Run a single Monte Carlo replication for all estimators."""
    d_true, phi, theta, rep_id, seed = args

    # Generate ARFIMA(1,d,1) data with ARMA(1,1) short-run dynamics
    x = arfima_arma(n_obs, d_true, phi=phi, theta=theta, sigma=1.0, seed=seed, burnin=0)

    results = {}

    # LW estimator
    lw = LW(taper='none')
    try:
        res = lw.estimate(x, m=m, bounds=bounds, verbose=False)
        results['LW'] = res['d_hat']
    except Exception:
        results['LW'] = np.nan

    # Velasco
    lw_v = LW(taper='kolmogorov')
    try:
        res = lw_v.estimate(x, m=m, bounds=bounds, verbose=False)
        results['V'] = res['d_hat']
    except Exception:
        results['V'] = np.nan

    # Hurvich-Chen
    lw_hc = LW(taper='hc')
    try:
        res = lw_hc.estimate(x, m=m, bounds=bounds, verbose=False)
        results['HC'] = res['d_hat']
    except Exception:
        results['HC'] = np.nan

    # ELW estimator
    elw = ELW()
    try:
        res = elw.estimate(x, m=m, bounds=bounds, verbose=False)
        results['ELW'] = res['d_hat']
    except Exception:
        results['ELW'] = np.nan

    # Two-step ELW
    elw2 = TwoStepELW()
    try:
        res = elw2.estimate(x, m=m, bounds=bounds, trend_order=0, verbose=False)
        results['2ELW'] = res['d_hat']
    except Exception:
        results['2ELW'] = np.nan

    # LWLFC (Hou-Perron)
    lwlfc = LWLFC()
    try:
        lwlfc.fit(x, m=m)
        results['LWLFC'] = lwlfc.d_hat_
    except Exception:
        results['LWLFC'] = np.nan

    return results


def main():
    """Main function to run the Monte Carlo simulation."""
    print("Estimator Comparison under ARMA(1,1) Short-Run Dynamics")
    print("=======================================================")
    print(f"Sample size: n = {n_obs}")
    print(f"Number of frequencies: m = n^{alpha} = {m}")
    print(f"Replications: {mc_reps}")
    print(f"d values: {d_list}")
    print(f"(phi, theta) configs: {arma_configs}")
    print(f"Estimators: {estimator_names}")
    print()

    # Prepare all simulation tasks
    tasks = []
    for ci, (phi, theta) in enumerate(arma_configs):
        for d_true in d_list:
            for rep in range(mc_reps):
                seed = (seed_base +
                        ci * len(d_list) * mc_reps +
                        d_list.index(d_true) * mc_reps +
                        rep)
                tasks.append((d_true, phi, theta, rep, seed))

    print("Running Monte Carlo simulations...")

    # Run simulations in parallel
    with Pool() as pool:
        all_results = pool.map(run_single_rep, tasks)

    # Organize results
    organized_results = {}
    task_idx = 0
    for phi, theta in arma_configs:
        for d_true in d_list:
            key = (d_true, phi, theta)
            organized_results[key] = {est: [] for est in estimator_names}

            for rep in range(mc_reps):
                rep_results = all_results[task_idx]
                for est_name, est_value in rep_results.items():
                    organized_results[key][est_name].append(est_value)
                task_idx += 1

    # Calculate statistics and prepare results
    results_list = []
    for (d_true, phi, theta), estimates_dict in organized_results.items():
        for est_name, estimates in estimates_dict.items():
            estimates_array = np.array(estimates)
            valid_estimates = estimates_array[~np.isnan(estimates_array)]

            if len(valid_estimates) > 0:
                bias = np.mean(valid_estimates) - d_true
                mse = np.mean((valid_estimates - d_true) ** 2)
            else:
                bias = mse = np.nan

            results_list.append({
                'd': d_true,
                'phi': phi,
                'theta': theta,
                'estimator': est_name,
                'bias': bias,
                'mse': mse,
                'n_valid': len(valid_estimates)
            })

    # Convert to DataFrame for easy manipulation
    results_df = pd.DataFrame(results_list)

    # Create results table (index by d, phi, theta)
    bias_pivot = results_df.pivot_table(values='bias', index=['d', 'phi', 'theta'], columns='estimator').round(4)
    mse_pivot = results_df.pivot_table(values='mse', index=['d', 'phi', 'theta'], columns='estimator').round(4)

    # Markdown table (config panels, ordered as arma_configs)
    headers = ['d', 'phi', 'theta'] + [f'{est}_b' for est in estimator_names] + [f'{est}_mse' for est in estimator_names]
    header_line = "| " + " | ".join(f"{h:>8}" for h in headers) + " |"
    separator_line = "|" + "|".join("-" * 10 for _ in headers) + "|"

    print()
    print(header_line)
    print(separator_line)

    for phi, theta in arma_configs:
        for d_val in d_list:
            bias_row = bias_pivot.loc[(d_val, phi, theta)]
            mse_row = mse_pivot.loc[(d_val, phi, theta)]
            values = [f"{d_val:>8.1f}", f"{phi:>8.1f}", f"{theta:>8.1f}"]
            for est in estimator_names:
                values.append(f"{bias_row[est]:>8.3f}" if est in bias_row.index else f"{'---':>8}")
            for est in estimator_names:
                values.append(f"{mse_row[est]:>8.3f}" if est in mse_row.index else f"{'---':>8}")
            print("| " + " | ".join(values) + " |")
    print()

    # Generate LaTeX table
    latex_table = """\\begin{table}[tp]
\\centering
\\begin{threeparttable}
\\caption{Estimator Comparison under ARMA(1,1) Short-Run Dynamics}
\\label{tab:mc_comp_arma11}
\\scriptsize
\\begin{tabular}{ccc|rrrrrr|rrrrrr}
\\toprule
&  &  & \\multicolumn{6}{c|}{Bias} & \\multicolumn{6}{c}{MSE} \\\\
\\cmidrule(lr){4-9} \\cmidrule(lr){10-15}
$d$ & $\\phi$ & $\\theta$ & LW & V & HC & ELW & 2ELW & LWLFC & LW & V & HC & ELW & 2ELW & LWLFC \\\\
\\midrule
"""

    for phi, theta in arma_configs:
        for d_val in d_list:
            bias_row = bias_pivot.loc[(d_val, phi, theta)]
            mse_row = mse_pivot.loc[(d_val, phi, theta)]
            latex_table += f"{d_val:4.1f} & {phi:3.1f} & {theta:4.1f} "

            # Bias columns
            for est in estimator_names:
                if est in bias_row.index:
                    latex_table += f"& {bias_row[est]:7.3f} "
                else:
                    latex_table += "& --- "

            # MSE columns
            for est in estimator_names:
                if est in mse_row.index:
                    latex_table += f"& {format_mse_latex(mse_row[est])} "
                else:
                    latex_table += "& --- "

            latex_table += "\\\\\n"

        latex_table += "\\midrule\n"

    # Drop the trailing group separator so \bottomrule is not doubled
    latex_table = latex_table.removesuffix("\\midrule\n")
    latex_table += f"""\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\footnotesize
\\item Notes: Monte Carlo results for $\\ARFIMA(1,d,1)$ processes with $n={n_obs}$, $m={m}$, {mc_reps:,} replications.
    Shaded cells indicate $\\text{{MSE}} > {MSE_THRESHOLD:.2f}$.
\\item LW = Local Whittle, V = Velasco (Kolmogorov), HC = Hurvich-Chen, ELW = Exact Local Whittle, 2ELW = Two-step ELW, LWLFC = Local Whittle with Low Frequency Contamination.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
"""

    # Save LaTeX table
    with open('tables/mc_comprehensive_arma1.tex', 'w') as f:
        f.write(latex_table)

    print("LaTeX table saved to: tables/mc_comprehensive_arma1.tex")


if __name__ == '__main__':
    main()
