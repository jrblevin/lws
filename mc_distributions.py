#!/usr/bin/env python3
"""
Sampling Distributions of LW Estimators

Extension of Shimotsu and Phillips (2005) Figure 1 showing finite sample
distributions of all local Whittle estimators across multiple d values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from scipy.stats import gaussian_kde

from pyelw import LW, ELW, TwoStepELW
from pyelw.simulate import arfima
from common import ESTIMATOR_COLORS

# Settings for distribution analysis
n_obs = 500
d_list = [-1.2, -0.4, 0.0, 0.4, 1.0, 1.6]
phi = 0.0
mu = 0.0
mc_reps = 10000
alpha = 0.65
m = int(n_obs**alpha)
bounds = (-4.0, 4.0)
seed_base = 42

estimator_names = ['LW', 'V', 'HC', 'ELW', '2ELW']


def run_single_rep(args):
    d_true, phi, rep_id, seed = args

    # Generate ARFIMA(1,d,0) data
    x = arfima(n_obs, d_true, phi=phi, sigma=1.0, seed=seed, burnin=0)

    # Add mean mu
    x = x + mu

    results = {}

    # LW estimator
    lw = LW(taper='none', bounds=bounds)
    try:
        res = lw.estimate(x, m=m, verbose=False)
        results['LW'] = res['d_hat']
    except Exception:
        results['LW'] = np.nan

    # Velasco (Kolmogorov) taper
    lw_v = LW(taper='kolmogorov', bounds=bounds)
    try:
        res = lw_v.estimate(x, m=m, verbose=False)
        results['V'] = res['d_hat']
    except Exception:
        results['V'] = np.nan

    # Hurvich-Chen taper
    lw_hc = LW(taper='hc', bounds=bounds)
    try:
        res = lw_hc.estimate(x, m=m, verbose=False)
        results['HC'] = res['d_hat']
    except Exception:
        results['HC'] = np.nan

    # ELW estimator
    elw = ELW(bounds=bounds)
    try:
        res = elw.estimate(x, m=m, verbose=False)
        results['ELW'] = res['d_hat']
    except Exception:
        results['ELW'] = np.nan

    # Two-step ELW
    elw2 = TwoStepELW(bounds=bounds, trend_order=0)
    try:
        res = elw2.estimate(x, m=m, verbose=False)
        results['2ELW'] = res['d_hat']
    except Exception:
        results['2ELW'] = np.nan

    return results


def generate_distribution_data():
    print("Generating Monte Carlo data for distribution plots...")
    print(f"Sample size: n = {n_obs}")
    print(f"Bandwidth: m = {m}")
    print(f"Replications: {mc_reps}")
    print(f"d values: {d_list}")
    print(f"phi: {phi}")

    start_time = time.time()

    # Prepare simulation tasks
    tasks = []
    for d_true in d_list:
        for rep in range(mc_reps):
            seed = seed_base + d_list.index(d_true) * mc_reps + rep
            tasks.append((d_true, phi, rep, seed))

    print("Running simulations in parallel...")

    # Run simulations
    with Pool() as pool:
        all_results = pool.map(run_single_rep, tasks)

    # Organize results
    results_list = []
    task_idx = 0
    for d_true in d_list:
        for rep in range(mc_reps):
            rep_results = all_results[task_idx]
            for est_name, est_value in rep_results.items():
                results_list.append({
                    'd': d_true,
                    'phi': phi,
                    'estimator': est_name,
                    'estimate': est_value,
                    'rep': rep
                })
            task_idx += 1

    df = pd.DataFrame(results_list)

    elapsed = time.time() - start_time
    print(f"Data generation completed in {elapsed:.1f} seconds")

    return df


def create_distribution_plots(data):
    nrows, ncols = 3, 2
    figsize = (12, 12)

    # Set up the plot
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    # Plot each d value in a separate subplot
    for i, d_val in enumerate(d_list):
        ax = axes[i]

        # Filter data for this d value
        d_data = data[data['d'] == d_val]

        # Plot distribution for each estimator using KDE
        for est_name in estimator_names:
            est_data = d_data[d_data['estimator'] == est_name]['estimate']
            est_data = est_data.dropna()

            # Create kernel density estimate
            kde = gaussian_kde(est_data)

            # Create x-axis for smooth curve
            x_min, x_max = d_val - 0.8, d_val + 0.8
            x_smooth = np.linspace(x_min, x_max, 200)
            density = kde(x_smooth)

            # Plot density curve
            ax.plot(x_smooth, density,
                    label=est_name, color=ESTIMATOR_COLORS.get(est_name, 'gray'),
                    linewidth=1.5, alpha=0.8)

        # Add true value line
        ax.axvline(d_val, color='black', linestyle='--', linewidth=1, label='True $d$')

        # Formatting
        ax.set_title(f'$d = {d_val}$', fontsize=14)
        if i == 1:
            ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots if any
    for j in range(len(d_list), len(axes)):
        axes[j].set_visible(False)

    # Overall formatting
    plt.tight_layout()

    # Save the plot
    plt.savefig('figures/mc_distributions.pdf', dpi=300, bbox_inches='tight')
    print("Distribution plots saved to figures/mc_distributions.pdf")

    return fig


def main():
    # Generate data
    data = generate_distribution_data()

    # Create distribution plots
    fig = create_distribution_plots(data)


if __name__ == "__main__":
    main()
