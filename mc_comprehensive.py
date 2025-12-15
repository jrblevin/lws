#!/usr/bin/env python3
"""
Comprehensive Estimator Comparison

Monte Carlo comparison of Local Whittle estimators with ARFIMA(1,d,0) data.
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool

from pyelw import LW, ELW, TwoStepELW
from pyelw.simulate import arfima
from common import format_mse_latex, MSE_THRESHOLD

# Settings from updated outline
n_obs = 500
d_list = [-2.2, -1.8, -1.2, -0.6, -0.3, 0.0, 0.3, 0.6, 1.2, 1.8, 2.2]
phi_list = [0.0, 0.5, 0.8]
mc_reps = 10000
alpha = 0.65
m = int(n_obs**alpha)
bounds = (-4.0, 4.0)
seed_base = 42

estimator_names = ['LW', 'V', 'HC', 'ELW', '2ELW']


def run_single_rep(args):
    """Run a single Monte Carlo replication for all estimators."""
    d_true, phi, rep_id, seed = args

    # Generate ARFIMA(1,d,0) data - match Shimotsu-Phillips exactly
    x = arfima(n_obs, d_true, phi=phi, sigma=1.0, seed=seed, burnin=0)

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

    return results


def main():
    """Main function to run the Monte Carlo simulation."""
    print("Comprehensive Estimator Comparison")
    print("==================================")
    print(f"Sample size: n = {n_obs}")
    print(f"Number of frequencies: m = n^{alpha} = {m}")
    print(f"Replications: {mc_reps}")
    print(f"d values: {d_list}")
    print(f"rho values: {phi_list}")
    print(f"Estimators: {estimator_names}")
    print()

    # Prepare all simulation tasks
    tasks = []
    for phi in phi_list:
        for d_true in d_list:
            for rep in range(mc_reps):
                seed = (seed_base +
                        phi_list.index(phi) * len(d_list) * mc_reps +
                        d_list.index(d_true) * mc_reps +
                        rep)
                tasks.append((d_true, phi, rep, seed))

    print("Running Monte Carlo simulations...")

    # Run simulations in parallel
    with Pool() as pool:
        all_results = pool.map(run_single_rep, tasks)

    # Organize results
    organized_results = {}
    task_idx = 0
    for phi in phi_list:
        for d_true in d_list:
            key = (d_true, phi)
            organized_results[key] = {est: [] for est in estimator_names}

            for rep in range(mc_reps):
                rep_results = all_results[task_idx]
                for est_name, est_value in rep_results.items():
                    organized_results[key][est_name].append(est_value)
                task_idx += 1

    # Calculate statistics and prepare results
    results_list = []
    for (d_true, phi), estimates_dict in organized_results.items():
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
                'estimator': est_name,
                'bias': bias,
                'mse': mse,
                'n_valid': len(valid_estimates)
            })

    # Convert to DataFrame for easy manipulation
    results_df = pd.DataFrame(results_list)

    # Create results table
    bias_pivot = results_df.pivot_table(values='bias', index=['d', 'phi'], columns='estimator').round(4)
    mse_pivot = results_df.pivot_table(values='mse', index=['d', 'phi'], columns='estimator').round(4)

    # Create dataframe with proper ordering (phi first, then d)
    combined_results = []
    for phi_val in phi_list:
        for d_val in d_list:
            row_data = {'d': d_val, 'phi': phi_val}

            # Add bias columns
            for est in estimator_names:
                if est in bias_pivot.columns:
                    row_data[f'{est}_bias'] = bias_pivot.loc[(d_val, phi_val), est]
                else:
                    row_data[f'{est}_bias'] = np.nan

            # Add MSE columns
            for est in estimator_names:
                if est in mse_pivot.columns:
                    row_data[f'{est}_mse'] = mse_pivot.loc[(d_val, phi_val), est]
                else:
                    row_data[f'{est}_mse'] = np.nan

            combined_results.append(row_data)

    combined_df = pd.DataFrame(combined_results)

    # Reorder columns for better display
    bias_cols = [f'{est}_bias' for est in estimator_names]
    mse_cols = [f'{est}_mse' for est in estimator_names]
    column_order = ['d', 'phi'] + bias_cols + mse_cols
    combined_df = combined_df[column_order]

    # Create Markdown table
    rounded_df = combined_df.round(4)

    # Header row
    headers = ['d', 'rho'] + [f'{est}_b' for est in estimator_names] + [f'{est}_mse' for est in estimator_names]
    header_line = "| " + " | ".join(f"{h:>8}" for h in headers) + " |"
    separator_line = "|" + "|".join("-" * 10 for _ in headers) + "|"

    print()
    print(header_line)
    print(separator_line)

    # Data rows
    for _, row in rounded_df.iterrows():
        values = [f"{row['d']:>8.1f}", f"{row['phi']:>8.1f}"]
        for est in estimator_names:
            bias_val = row[f'{est}_bias']
            mse_val = row[f'{est}_mse']
            values.append(f"{bias_val:>8.3f}" if not pd.isna(bias_val) else f"{'---':>8}")
        for est in estimator_names:
            mse_val = row[f'{est}_mse']
            values.append(f"{mse_val:>8.3f}" if not pd.isna(mse_val) else f"{'---':>8}")

        data_line = "| " + " | ".join(values) + " |"
        print(data_line)
    print()

    # Generate LaTeX table
    latex_table = f"""\\begin{{table}}[htbp]
\\centering
\\begin{{threeparttable}}
\\caption{{Comprehensive Estimator Comparison}}
\\label{{tab:mc_comprehensive}}
\\footnotesize
\\begin{{tabular}}{{cc|ccccc|ccccc}}
\\toprule
&  & \\multicolumn{{5}}{{c|}}{{Bias}} & \\multicolumn{{5}}{{c}}{{MSE}} \\\\
\\cmidrule(lr){{3-7}} \\cmidrule(lr){{8-12}}
$d$ & $\\rho$ & LW & V & HC & ELW & 2ELW & LW & V & HC & ELW & 2ELW \\\\
\\midrule
"""

    for phi_val in phi_list:
        for d_val in d_list:
            bias_row = bias_pivot.loc[(d_val, phi_val)]
            mse_row = mse_pivot.loc[(d_val, phi_val)]
            latex_table += f"{d_val:4.1f} & {phi_val:3.1f} "

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

    latex_table += f"""\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\footnotesize
\\item Notes: Monte Carlo results for $\\ARFIMA(1,d,0)$ processes with $n={n_obs}$, $m={m}$, {mc_reps:,} replications.
    Shaded cells indicate $\\text{{MSE}} > {MSE_THRESHOLD:.2f}$.
\\item LW = Local Whittle, V = Velasco (Kolmogorov), HC = Hurvich-Chen, ELW = Exact Local Whittle, 2ELW = Two-step ELW.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
"""

    # Save LaTeX table
    with open('tables/mc_comprehensive.tex', 'w') as f:
        f.write(latex_table)

    print("LaTeX table saved to: tables/mc_comprehensive.tex")


if __name__ == '__main__':
    main()
