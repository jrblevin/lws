#!/usr/bin/env python3
"""
Estimator Comparison under Heavy-Tailed and GARCH Innovations

Monte Carlo comparison of local Whittle estimators with ARFIMA(0,d,0) data
generated from non-Gaussian innovations. This is a companion to
mc_comprehensive.py, which uses Gaussian innovations with AR(1) short-run
dynamics. The design, estimators, parameter grid, bandwidth, and seed scheme
match mc_comprehensive.py exactly, but the AR(1) coefficient rho is replaced
by the innovation distribution: Gaussian (baseline), t(5), or GARCH(1,1).
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool

from pyelw import LW, ELW, TwoStepELW, LWLFC
from pyelw.simulate import fracdiff
from common import format_mse_latex, MSE_THRESHOLD

# Settings (match mc_comprehensive.py, with dist in place of rho)
n_obs = 500
d_list = [-2.2, -1.8, -1.2, -0.6, -0.3, 0.0, 0.3, 0.6, 1.2, 1.8, 2.2]
dist_list = ['gaussian', 't5', 'garch']
mc_reps = 10000
alpha = 0.65
m = int(n_obs**alpha)
bounds = (-4.0, 4.0)
seed_base = 42

# GARCH(1,1) parameters: unit unconditional variance, persistence 0.9
garch_omega = 0.1
garch_alpha = 0.1
garch_beta = 0.8
garch_burnin = 500

estimator_names = ['LW', 'V', 'HC', 'ELW', '2ELW', 'LWLFC']

dist_labels = {
    'gaussian': 'Gaussian innovations (baseline)',
    't5': 'Student-$t(5)$ innovations',
    'garch': 'GARCH(1,1) innovations',
}


def innovations(dist, n, seed):
    """Generate n mean-zero, unit-variance innovations from the given family."""
    np.random.seed(seed)

    if dist == 'gaussian':
        return np.random.normal(0, 1.0, n)

    if dist == 't5':
        # Rescale to unit variance
        return np.random.standard_t(5, n) / np.sqrt(5.0 / 3.0)

    if dist == 'garch':
        n_total = n + garch_burnin
        eta = np.random.normal(0, 1.0, n_total)
        eps = np.empty(n_total)
        sigma2 = garch_omega / (1.0 - garch_alpha - garch_beta)  # = 1.0
        eps[0] = np.sqrt(sigma2) * eta[0]
        for t in range(1, n_total):
            sigma2 = (garch_omega
                      + garch_alpha * eps[t-1]**2
                      + garch_beta * sigma2)
            eps[t] = np.sqrt(sigma2) * eta[t]
        return eps[garch_burnin:]

    raise ValueError(f"Unknown innovation distribution: {dist}")


def run_single_rep(args):
    """Run a single Monte Carlo replication for all estimators."""
    d_true, dist, rep_id, seed = args

    # Generate ARFIMA(0,d,0) data with the given innovation distribution
    eps = innovations(dist, n_obs, seed)
    if abs(d_true) < 1e-13:
        x = eps
    else:
        x = fracdiff(eps, -d_true)

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
    print("Estimator Comparison under Heavy-Tailed and GARCH Innovations")
    print("=============================================================")
    print()
    print(f"Sample size: n = {n_obs}")
    print(f"Number of frequencies: m = n^{alpha} = {m}")
    print(f"Replications: {mc_reps}")
    print(f"d values: {d_list}")
    print(f"Innovation distributions: {dist_list}")
    print(f"GARCH(1,1) parameters: omega={garch_omega}, alpha={garch_alpha}, beta={garch_beta}")
    print(f"Estimators: {estimator_names}")
    print()

    # Prepare all simulation tasks
    tasks = []
    for dist in dist_list:
        for d_true in d_list:
            for rep in range(mc_reps):
                seed = (seed_base +
                        dist_list.index(dist) * len(d_list) * mc_reps +
                        d_list.index(d_true) * mc_reps +
                        rep)
                tasks.append((d_true, dist, rep, seed))

    print("Running Monte Carlo simulations...")

    # Run simulations in parallel
    with Pool() as pool:
        all_results = pool.map(run_single_rep, tasks)

    # Organize results
    organized_results = {}
    task_idx = 0
    for dist in dist_list:
        for d_true in d_list:
            key = (d_true, dist)
            organized_results[key] = {est: [] for est in estimator_names}

            for rep in range(mc_reps):
                rep_results = all_results[task_idx]
                for est_name, est_value in rep_results.items():
                    organized_results[key][est_name].append(est_value)
                task_idx += 1

    # Calculate statistics and prepare results
    results_list = []
    for (d_true, dist), estimates_dict in organized_results.items():
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
                'dist': dist,
                'estimator': est_name,
                'bias': bias,
                'mse': mse,
                'n_valid': len(valid_estimates)
            })

    # Convert to DataFrame for easy manipulation
    results_df = pd.DataFrame(results_list)

    # Create results table
    bias_pivot = results_df.pivot_table(values='bias', index=['d', 'dist'], columns='estimator').round(4)
    mse_pivot = results_df.pivot_table(values='mse', index=['d', 'dist'], columns='estimator').round(4)

    # Print Markdown table
    headers = ['d', 'dist'] + [f'{est}_b' for est in estimator_names] + [f'{est}_mse' for est in estimator_names]
    header_line = "| " + " | ".join(f"{h:>8}" for h in headers) + " |"
    separator_line = "|" + "|".join("-" * 10 for _ in headers) + "|"

    print()
    print(header_line)
    print(separator_line)

    for dist in dist_list:
        for d_val in d_list:
            bias_row = bias_pivot.loc[(d_val, dist)]
            mse_row = mse_pivot.loc[(d_val, dist)]
            values = [f"{d_val:>8.1f}", f"{dist:>8}"]
            for est in estimator_names:
                bias_val = bias_row.get(est, np.nan)
                values.append(f"{bias_val:>8.3f}" if not pd.isna(bias_val) else f"{'---':>8}")
            for est in estimator_names:
                mse_val = mse_row.get(est, np.nan)
                values.append(f"{mse_val:>8.3f}" if not pd.isna(mse_val) else f"{'---':>8}")

            data_line = "| " + " | ".join(values) + " |"
            print(data_line)
    print()

    # Generate LaTeX table
    latex_table = """\\begin{table}[tp]
\\centering
\\begin{threeparttable}
\\caption{Estimator Comparison under Heavy-Tailed and GARCH Innovations}
\\label{tab:mc_comp_heavy}
\\scriptsize
\\begin{tabular}{c|rrrrrr|rrrrrr}
\\toprule
& \\multicolumn{6}{c|}{Bias} & \\multicolumn{6}{c}{MSE} \\\\
\\cmidrule(lr){2-7} \\cmidrule(lr){8-13}
$d$ & LW & V & HC & ELW & 2ELW & LWLFC & LW & V & HC & ELW & 2ELW & LWLFC \\\\
\\midrule
"""

    for dist in dist_list:
        # Omit Gaussian baseline from LaTeX table
        if dist == 'gaussian':
            continue
        latex_table += (f"\\multicolumn{{13}}{{l}}{{\\textit{{{dist_labels[dist]}}}}} \\\\\n")
        for d_val in d_list:
            bias_row = bias_pivot.loc[(d_val, dist)]
            mse_row = mse_pivot.loc[(d_val, dist)]
            latex_table += f"{d_val:4.1f} "

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
\\item Notes: Monte Carlo results for $\\ARFIMA(0,d,0)$ processes with $n={n_obs}$, $m={m}$, {mc_reps:,} replications,
    and mean-zero, unit-variance innovations $\\varepsilon_t$.
    Student-$t(5)$: $\\varepsilon_t = \\eta_t / \\sqrt{{\\nf{{5}}{{3}}}}$ with
    $\\eta_t \\iidsim t(5)$ (kurtosis 9).
    GARCH(1,1): $\\varepsilon_t = \\sigma_t \\eta_t$ with $\\eta_t \\iidsim \\Normal(0,1)$ and
    $\\sigma_t^2 = \\omega + \\alpha \\varepsilon_{{t-1}}^2 + \\beta \\sigma_{{t-1}}^2$,
    $(\\omega, \\alpha, \\beta) = ({garch_omega}, {garch_alpha}, {garch_beta})$,
    implying unit unconditional variance and kurtosis $\\approx 3.35$.
    Shaded cells indicate $\\text{{MSE}} > {MSE_THRESHOLD:.2f}$.
\\item LW = Local Whittle, V = Velasco (Kolmogorov), HC = Hurvich-Chen, ELW = Exact Local Whittle, 2ELW = Two-step ELW, LWLFC = Local Whittle with Low Frequency Contamination.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
"""

    # Save LaTeX table
    with open('tables/mc_comprehensive_heavy.tex', 'w') as f:
        f.write(latex_table)

    print("LaTeX table saved to: tables/mc_comprehensive_heavy.tex")


if __name__ == '__main__':
    main()
