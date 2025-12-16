#!/usr/bin/env python3
"""
Robustness to Time Trend

Comparison showing how estimators perform under trend contamination.
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool

from pyelw import LW, ELW, TwoStepELW, LWLFC
from common import format_mse_latex, format_mse_ratio_latex, MSE_THRESHOLD, MSE_RATIO_THRESHOLD
from pyelw.simulate import arfima

# Settings from updated outline
n_obs = 500
d_list = [-2.2, -1.8, -1.2, -0.6, -0.3, 0.0, 0.3, 0.6, 1.2, 1.8, 2.2]
phi = 0.0
trend_list = [0.0, 0.05]
trend_correction_methods = ['none', 'detrend']
mc_reps = 10000
alpha = 0.65
m = int(n_obs**alpha)
bounds = (-4.0, 4.0)
seed_base = 42

estimator_names = ['LW', 'V', 'HC', 'ELW', '2ELW', 'LWLFC']


def run_single_rep(args):
    """Run a single Monte Carlo replication for all estimators."""
    d_true, phi, trend_slope, trend_correction, rep_id, seed = args

    # Generate ARFIMA(1,d,0) data
    x = arfima(n_obs, d_true, phi=phi, sigma=1.0, seed=seed, burnin=0)

    # Add linear trend if specified
    if trend_slope != 0:
        t = np.arange(1, n_obs + 1)
        x = x + trend_slope * t

    # Save original data with trend for 2ELW
    x_orig = x.copy()

    # Apply trend correction if specified
    if trend_correction == 'detrend':
        # Remove linear trend by OLS detrending
        t = np.arange(1, n_obs + 1)
        # Simple linear detrending: x_detrended = x - (a + b*t)
        A = np.vstack([np.ones(n_obs), t]).T
        coeffs = np.linalg.lstsq(A, x, rcond=None)[0]
        x = x - (coeffs[0] + coeffs[1] * t)

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

    # Two-step ELW (always uses original data with internal linear detrending)
    elw2 = TwoStepELW()
    try:
        # Use original data with trend, apply internal linear detrending
        res = elw2.estimate(x_orig, m=m, bounds=bounds, trend_order=1, verbose=False)
        results['2ELW'] = res['d_hat']
    except Exception:
        results['2ELW'] = np.nan

    # LWLFC (Hou-Perron) - robust to low frequency contamination including trends
    lwlfc = LWLFC()
    try:
        lwlfc.fit(x, m=m)
        results['LWLFC'] = lwlfc.d_hat_
    except Exception:
        results['LWLFC'] = np.nan

    return results


def main():
    """Main function to run the Monte Carlo simulation."""
    print("Robustness to Time Trend")
    print("========================")
    print(f"Sample size: n = {n_obs}")
    print(f"Number of frequencies: m = n^{alpha} = {m}")
    print(f"Replications: {mc_reps}")
    print(f"d values: {d_list}")
    print(f"rho: {phi} (fixed)")
    print(f"Trend slopes: {trend_list}")
    print(f"Trend correction methods: {trend_correction_methods}")
    print("Estimators: LW, V, HC, ELW, 2ELW")
    print()

    # Prepare all simulation tasks
    tasks = []
    for d_true in d_list:
        for trend_slope in trend_list:
            for trend_correction in trend_correction_methods:
                for rep in range(mc_reps):
                    # Same seed across trend_correction methods for fair comparison
                    seed = (seed_base +
                            d_list.index(d_true) * len(trend_list) * mc_reps +
                            trend_list.index(trend_slope) * mc_reps +
                            rep)
                    tasks.append((d_true, phi, trend_slope, trend_correction, rep, seed))

    print("Running Monte Carlo simulation...")

    # Run simulations in parallel
    with Pool() as pool:
        all_results = pool.map(run_single_rep, tasks)

    # Organize results
    organized_results = {}
    task_idx = 0
    for d_true in d_list:
        for trend_slope in trend_list:
            for trend_correction in trend_correction_methods:
                key = (d_true, trend_slope, trend_correction)
                organized_results[key] = {est: [] for est in estimator_names}

                for rep in range(mc_reps):
                    rep_results = all_results[task_idx]
                    for est_name, est_value in rep_results.items():
                        if est_name in organized_results[key]:
                            organized_results[key][est_name].append(est_value)
                    task_idx += 1

    # Calculate statistics and prepare results
    results_list = []
    for (d_true, trend_slope, trend_correction), estimates_dict in organized_results.items():
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
                'trend': trend_slope,
                'trend_correction': trend_correction,
                'estimator': est_name,
                'bias': bias,
                'mse': mse,
                'n_valid': len(valid_estimates)
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)

    # Create comparison table showing trend correction effects by panel
    print("## MSE Results by Trend Level and Correction Method")
    print()

    trend_clean = trend_list[0]
    trend_contam = trend_list[1]

    panel_names = {
        'none': 'Panel A: No Trend Correction',
        'detrend': 'Panel B: Linear Trend Correction (OLS Detrending)'
    }

    for trend_correction in trend_correction_methods:
        print(f"### {panel_names[trend_correction]}")
        print()

        # Filter results for this trend correction method
        panel_df = results_df[results_df['trend_correction'] == trend_correction]
        mse_pivot = panel_df.pivot_table(values='mse', index='d', columns=['trend', 'estimator']).round(4)

        # Table header
        print(f"|      | trend={trend_clean}                                     | trend={trend_contam}                                    |")
        print("|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|")
        print("|  d   |  LW   |   V   |  HC   |  ELW  | 2ELW  | LWLFC |  LW   |   V   |  HC   |  ELW  | 2ELW  | LWLFC |")
        print("|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|")

        for d_val in d_list:
            line = f"| {d_val:4.1f} |"
            # Clean values (no trend)
            for est in estimator_names:
                try:
                    val = mse_pivot.loc[d_val, (trend_clean, est)]
                    line += f" {val:5.3f} |"
                except Exception:
                    line += "  ---  |"
            # Contaminated values (with trend)
            for est in estimator_names:
                try:
                    val = mse_pivot.loc[d_val, (trend_contam, est)]
                    line += f" {val:5.3f} |"
                except Exception:
                    line += "  ---  |"
            print(line)
        print()

    # Calculate MSE ratios (contaminated/clean) by panel
    print(f"## MSE Degradation Ratios (trend={trend_contam} / trend={trend_clean})")
    print()

    for trend_correction in trend_correction_methods:
        print(f"### {panel_names[trend_correction]}")
        print()

        # Filter results for this trend correction method
        panel_df = results_df[results_df['trend_correction'] == trend_correction]
        clean_results = panel_df[panel_df['trend'] == trend_clean].set_index(['d', 'estimator'])['mse']
        contam_results = panel_df[panel_df['trend'] == trend_contam].set_index(['d', 'estimator'])['mse']

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

    # Generate combined LaTeX table with both panels
    latex_table = f"""\\begin{{table}}[t!]
\\centering
\\begin{{threeparttable}}
\\caption{{Robustness to Time Trend}}
\\label{{tab:mc_time_trend}}
\\footnotesize
\\begin{{tabular}}{{c|rrrrrr|rrrrrr}}
\\toprule
& \\multicolumn{{6}}{{c|}}{{Baseline MSE ($\\beta = {trend_clean}$)}} & \\multicolumn{{6}}{{c}}{{MSE Ratio ($\\beta = {trend_contam}$ / $\\beta = {trend_clean}$)}} \\\\
\\cmidrule(lr){{2-7}} \\cmidrule(lr){{8-13}}
$d$ & LW & V & HC & ELW & 2ELW & LWLFC & LW & V & HC & ELW & 2ELW & LWLFC \\\\
"""

    for panel_idx, trend_correction in enumerate(trend_correction_methods):
        panel_description = {
            'none': 'Trend correction: None',
            'detrend': 'Trend correction: linear OLS detrending'
        }[trend_correction]

        # Add panel separator (except for first panel)
        if panel_idx > 0:
            latex_table += "\\midrule\n"
        else:
            latex_table += "\\midrule\n"

        # Add panel label row
        latex_table += f"\\multicolumn{{13}}{{c}}{{\\textit{{{panel_description}}}}} \\\\\n\\midrule\n"

        # Filter results for this panel
        panel_df = results_df[results_df['trend_correction'] == trend_correction]
        clean_results = panel_df[panel_df['trend'] == trend_clean].set_index(['d', 'estimator'])['mse']
        contam_results = panel_df[panel_df['trend'] == trend_contam].set_index(['d', 'estimator'])['mse']

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

    # Format ARFIMA specification based on phi value
    if phi == 0:
        arfima_spec = "$\\ARFIMA(0,d,0)$"
    else:
        arfima_spec = f"$\\ARFIMA(1,d,0)$ with $\\rho={phi}$"

    latex_table += f"""\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\footnotesize
\\item Notes: MSE results for {arfima_spec}, n={n_obs}, m={m}, {mc_reps:,} replications.
    Shaded cells indicate $\\text{{MSE}} > {MSE_THRESHOLD:.2f}$ or $\\text{{MSE Ratio}} > {MSE_RATIO_THRESHOLD:.1f}$.
\\item LW = Local Whittle, V = Velasco (Kolmogorov), HC = Hurvich-Chen, ELW = Exact Local Whittle, 2ELW = Two-step ELW with linear detrending and adaptive mean estimation applied to original series, LWLFC = Local Whittle with Low Frequency Contamination.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
"""

    # Save combined LaTeX table
    with open('tables/mc_time_trend.tex', 'w') as f:
        f.write(latex_table)

    print("LaTeX table saved to: tables/mc_time_trend.tex")


if __name__ == '__main__':
    main()
