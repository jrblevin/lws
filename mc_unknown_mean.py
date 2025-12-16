#!/usr/bin/env python3
"""
Robustness to Unknown Mean

Comparison showing how estimators perform with nonzero mean.
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool

from pyelw import LW, ELW, TwoStepELW, LWLFC
from common import format_mse_latex, format_mse_ratio_latex, MSE_THRESHOLD, MSE_RATIO_THRESHOLD
from pyelw.simulate import arfima

# Settings from updated outline - configurable parameters
n_obs = 500
d_list = [-2.2, -1.8, -1.2, -0.6, -0.3, 0.0, 0.3, 0.6, 1.2, 1.8, 2.2]
phi = 0.0
mu_list = [0, 5]  # Zero vs. non-zero mean
mc_reps = 10000
alpha = 0.65
m = int(n_obs**alpha)
bounds = (-4.0, 4.0)
seed_base = 42

estimator_names = ['LW', 'V', 'HC', 'ELW', '2ELW', 'LWLFC']
mean_est_methods = ['none', 'mean', 'init']


def run_single_rep(args):
    """Run a single Monte Carlo replication for all estimators."""
    d_true, phi, mu, mean_est, rep_id, seed = args

    # Generate ARFIMA(1,d,0) data
    x = arfima(n_obs, d_true, phi=phi, sigma=1.0, seed=seed, burnin=0)

    # Add mean mu
    x = x + mu

    # Apply mean correction based on method
    if mean_est == 'mean':
        # Subtract sample mean
        x_corrected = x - np.mean(x)
    elif mean_est == 'init':
        # Subtract initial value
        x_corrected = x - x[0]
    else:  # mean_est == 'none'
        # No correction
        x_corrected = x

    results = {}

    # LW estimator
    lw = LW(taper='none')
    try:
        res = lw.estimate(x_corrected, m=m, bounds=bounds, verbose=False)
        results['LW'] = res['d_hat']
    except Exception:
        results['LW'] = np.nan

    # Velasco
    lw_v = LW(taper='kolmogorov')
    try:
        res = lw_v.estimate(x_corrected, m=m, bounds=bounds, verbose=False)
        results['V'] = res['d_hat']
    except Exception:
        results['V'] = np.nan

    # Hurvich-Chen
    lw_hc = LW(taper='hc')
    try:
        res = lw_hc.estimate(x_corrected, m=m, bounds=bounds, verbose=False)
        results['HC'] = res['d_hat']
    except Exception:
        results['HC'] = np.nan

    # ELW estimator
    elw = ELW()
    try:
        res = elw.estimate(x_corrected, m=m, bounds=bounds, verbose=False)
        results['ELW'] = res['d_hat']
    except Exception:
        results['ELW'] = np.nan

    # Two-step ELW (always does its own mean handling)
    elw2 = TwoStepELW()
    try:
        # Use original x for 2ELW since it handles mean internally
        res = elw2.estimate(x, m=m, bounds=bounds, trend_order=0, verbose=False)
        results['2ELW'] = res['d_hat']
    except Exception:
        results['2ELW'] = np.nan

    # LWLFC (Hou-Perron) - robust to low frequency contamination
    lwlfc = LWLFC()
    try:
        lwlfc.fit(x_corrected, m=m)
        results['LWLFC'] = lwlfc.d_hat_
    except Exception:
        results['LWLFC'] = np.nan

    return results


def main():
    print("Robustness to Unknown Mean")
    print("==========================")
    print(f"Sample size: n = {n_obs}")
    print(f"Number of frequencies: m = n^{alpha} = {m}")
    print(f"Replications: {mc_reps}")
    print(f"d values: {d_list}")
    print(f"rho: {phi} (fixed)")
    print(f"Mean values: {mu_list}")
    print(f"Mean estimation methods: {mean_est_methods}")
    print()

    # Prepare all simulation tasks
    tasks = []
    for d_true in d_list:
        for mu in mu_list:
            for mean_est in mean_est_methods:
                for rep in range(mc_reps):
                    # Same seed across mean_est methods for fair comparison
                    seed = (seed_base +
                            d_list.index(d_true) * len(mu_list) * mc_reps +
                            mu_list.index(mu) * mc_reps +
                            rep)
                    tasks.append((d_true, phi, mu, mean_est, rep, seed))

    print("Running Monte Carlo simulation...")

    # Run simulations in parallel
    with Pool() as pool:
        all_results = pool.map(run_single_rep, tasks)

    # Organize results
    organized_results = {}
    task_idx = 0
    for d_true in d_list:
        for mu in mu_list:
            for mean_est in mean_est_methods:
                key = (d_true, mu, mean_est)
                organized_results[key] = {est: [] for est in estimator_names}

                for rep in range(mc_reps):
                    rep_results = all_results[task_idx]
                    for est_name, est_value in rep_results.items():
                        organized_results[key][est_name].append(est_value)
                    task_idx += 1

    # Calculate statistics and prepare results
    results_list = []
    for (d_true, mu, mean_est), estimates_dict in organized_results.items():
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
                'mu': mu,
                'mean_est': mean_est,
                'estimator': est_name,
                'bias': bias,
                'mse': mse,
                'n_valid': len(valid_estimates)
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)

    # Create comparison table showing contamination effects by panel
    print("## MSE Results by Mean Level and Correction Method")
    print()

    mu_zero = mu_list[0]
    mu_pos = mu_list[1]

    panel_names = {
        'none': 'Panel A: No Mean Correction',
        'mean': 'Panel B: Sample Mean Correction (mu_hat = mean(X))',
        'init': 'Panel C: Initial Observation Correction (mu_hat = X_1)'
    }

    for mean_est in mean_est_methods:
        print(f"### {panel_names[mean_est]}")
        print()

        # Filter results for this mean estimation method
        panel_df = results_df[results_df['mean_est'] == mean_est]
        mse_pivot = panel_df.pivot_table(values='mse', index='d', columns=['mu', 'estimator']).round(4)

        # Table header
        print(f"|      | mu={mu_zero}                                          | mu={mu_pos}                                          |")
        print("|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|")
        print("|  d   |  LW   |   V   |  HC   |  ELW  | 2ELW  | LWLFC |  LW   |   V   |  HC   |  ELW  | 2ELW  | LWLFC |")
        print("|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|")

        for d_val in d_list:
            line = f"| {d_val:4.1f} |"
            # Clean values
            for est in estimator_names:
                try:
                    val = mse_pivot.loc[d_val, (mu_zero, est)]
                    line += f" {val:5.3f} |"
                except Exception:
                    line += "  ---  |"
            # Contaminated values
            for est in estimator_names:
                try:
                    val = mse_pivot.loc[d_val, (mu_pos, est)]
                    line += f" {val:5.3f} |"
                except Exception:
                    line += "  ---  |"
            print(line)
        print()

    # Calculate MSE ratios (contaminated/clean) by panel
    print(f"## MSE Degradation Ratios (mu={mu_pos} / mu={mu_zero})")
    print()

    for mean_est in mean_est_methods:
        print(f"### {panel_names[mean_est]}")
        print()

        # Filter results for this mean estimation method
        panel_df = results_df[results_df['mean_est'] == mean_est]
        clean_results = panel_df[panel_df['mu'] == mu_zero].set_index(['d', 'estimator'])['mse']
        contam_results = panel_df[panel_df['mu'] == mu_pos].set_index(['d', 'estimator'])['mse']

        # Manual markdown table for ratios
        print("|  d   |    LW   |    V    |    HC   |   ELW   |   2ELW  |  LWLFC  |")
        print("|------|---------|---------|---------|---------|---------|---------|")
        for d_val in d_list:
            line = f"| {d_val:4.1f} |"
            for est in estimator_names:
                clean_val = clean_results.loc[(d_val, est)]
                contam_val = contam_results.loc[(d_val, est)]
                ratio = contam_val / clean_val
                line += f" {ratio:7.2f} |"
            print(line)
        print()

    # Generate combined LaTeX table with all panels
    latex_table = f"""\\begin{{table}}[tbp]
\\centering
\\begin{{threeparttable}}
\\caption{{Robustness to Unknown Mean}}
\\label{{tab:mc_unknown_mean}}
\\footnotesize
\\begin{{tabular}}{{c|rrrrrr|rrrrrr}}
\\toprule
 & \\multicolumn{{6}}{{c|}}{{Baseline MSE ($\\mu = {mu_zero}$)}} & \\multicolumn{{6}}{{c}}{{MSE Ratio ($\\mu = {mu_pos}$ / $\\mu = {mu_zero}$)}} \\\\
\\cmidrule(lr){{2-7}} \\cmidrule(lr){{8-13}}
$d$ & LW & V & HC & ELW & 2ELW & LWLFC & LW & V & HC & ELW & 2ELW & LWLFC \\\\
"""

    for panel_idx, mean_est in enumerate(mean_est_methods):
        panel_description = {
            'none': 'Mean correction: None',
            'mean': r'Mean correction: $\hat{\mu} = \bar{X}$',
            'init': r'Mean correction: $\hat{\mu} = X_1$'
        }[mean_est]

        # Add panel separator (except for first panel)
        if panel_idx > 0:
            latex_table += "\\midrule\n"
        else:
            latex_table += "\\midrule\n"

        # Add panel label row
        latex_table += f"\\multicolumn{{13}}{{c}}{{\\textit{{{panel_description}}}}} \\\\\n\\midrule\n"

        # Filter results for this panel
        panel_df = results_df[results_df['mean_est'] == mean_est]
        clean_results = panel_df[panel_df['mu'] == mu_zero].set_index(['d', 'estimator'])['mse']
        contam_results = panel_df[panel_df['mu'] == mu_pos].set_index(['d', 'estimator'])['mse']

        for d_val in d_list:
            latex_table += f"{d_val:3.1f} "

            # Baseline MSE values
            for est in estimator_names:
                try:
                    clean_val = clean_results.loc[(d_val, est)]
                    latex_table += f"& {format_mse_latex(clean_val)} "
                except Exception:
                    latex_table += "& -- "

            # MSE Ratios
            for est in estimator_names:
                try:
                    clean_val = clean_results.loc[(d_val, est)]
                    contam_val = contam_results.loc[(d_val, est)]
                    ratio_val = contam_val / clean_val
                    latex_table += f"& {format_mse_ratio_latex(ratio_val)} "
                except Exception:
                    latex_table += "& -- "

            latex_table += "\\\\\n"

    latex_table += f"""\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\footnotesize
\\item Notes: MSE results for $\\ARFIMA(0,d,0)$ with $n={n_obs}$, $m={m}$, {mc_reps:,} replications.
    Shaded cells indicate $\\text{{MSE}} > {MSE_THRESHOLD:.2f}$ or $\\text{{MSE Ratio}} > {MSE_RATIO_THRESHOLD:.1f}$.
\\item LW = Local Whittle, V = Velasco (Kolmogorov), HC = Hurvich-Chen, ELW = Exact Local Whittle, 2ELW = Two-step ELW with adaptive mean estimation applied to original series, LWLFC = Local Whittle with Low Frequency Contamination.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
"""

    # Save LaTeX table
    with open('tables/mc_unknown_mean.tex', 'w') as f:
        f.write(latex_table)

    print("LaTeX table saved to: tables/mc_unknown_mean.tex")


if __name__ == '__main__':
    main()
