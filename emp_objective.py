#!/usr/bin/env python3
"""
Objective Function Plots

Plots objective functions for all estimators using data from
Hurvich and Chen (2000).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyelw import LW, ELW, TwoStepELW
from common import ESTIMATOR_COLORS, ESTIMATOR_LINESTYLES

# U.S. Industrial Production data from Hurvich and Chen (2000)
# This series shows interesting estimator disagreement
SERIES = {
    'filename': 'data/indpro_us.dat',
    'name': 'U.S. Industrial Production',
    'transform': 'log',
    'm': 100,
}


def plot_objective_functions():
    # Set up the figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Common settings
    bounds = (-1.0, 3.0)
    d_grid = np.linspace(bounds[0], bounds[1], 200)

    # Load data
    print(f"Loading {SERIES['name']} data...")
    if SERIES['filename'].endswith('.csv'):
        data = pd.read_csv(SERIES['filename'])
        series = data[SERIES['column']].dropna().values
    else:
        series = np.loadtxt(SERIES['filename'])

    # Apply transform if specified
    if SERIES.get('transform') == 'log':
        series = np.log(series)
    elif SERIES.get('transform') == 'diff-log':
        series = np.log(series)
        series = np.diff(series)

    n = len(series)
    m = SERIES['m']

    print(f"{SERIES['name']}: n={n}, m={m}")

    # Plot objective functions
    plot_series_objectives(series, m, d_grid, ax, SERIES['name'], bounds)

    # Overall formatting
    plt.tight_layout()
    plt.savefig('figures/emp_objective.pdf', dpi=300, bbox_inches='tight')
    print("Objective function plots saved to figures/emp_objective.pdf")


def plot_series_objectives(series, m, d_grid, ax, title, bounds):
    # Initialize estimators
    estimators = {
        'LW': LW(bounds=bounds, taper='none'),
        'V': LW(bounds=bounds, taper='kolmogorov'),
        'HC': LW(bounds=bounds, taper='hc'),
        'ELW': ELW(bounds=bounds),
        'ELW-DM': ELW(bounds=bounds, mean_est='mean'),
        '2ELW': TwoStepELW(bounds=bounds, trend_order=0),
    }

    # Calculate objective functions
    objectives = {}

    print(f"Evaluating objective functions:")
    for est_name, estimator in estimators.items():
        print(f" - {est_name}...")
        obj_values = []

        # Prepare data with the correct taper for each estimator
        if est_name == 'LW':
            prepared_data = estimator.prepare_data(series, m, taper='none')
        elif est_name == 'V':
            prepared_data = estimator.prepare_data(series, m, taper='kolmogorov')
        elif est_name == 'HC':
            prepared_data = estimator.prepare_data(series, m, taper='hc')

        for d in d_grid:
            if est_name == 'LW':
                obj_val = estimator.objective(d, prepared_data)
            elif est_name == 'V':
                obj_val = estimator.objective_velasco(d, prepared_data)
            elif est_name == 'HC':
                # HC works on differenced data, so subtract diff=1 from d
                obj_val = estimator.objective_hc(d - 1, prepared_data)
            elif est_name == 'ELW-DM':
                # ELW with demeaned data
                series_demean = series - np.mean(series)
                obj_val = estimator.objective(d, series_demean, m)
            else:
                # ELW and 2ELW use different signature
                obj_val = estimator.objective(d, series, m)
            obj_values.append(obj_val)

        objectives[est_name] = np.array(obj_values)

    # Find actual estimates for markers
    estimates = {}
    print()
    print('Computing estimates:')
    for est_name, estimator in estimators.items():
        res = estimator.fit(series, m=m, verbose=False)
        estimates[est_name] = res.d_hat_
        print(f" - {est_name}: d_hat = {res.d_hat_:.2f}")

    print()
    print('Plotting objective functions...')
    # Plot objective functions with markers at minima
    for est_name in estimators:
        obj_vals = objectives[est_name]

        # Remove NaNs for plotting
        valid_mask = ~np.isnan(obj_vals)
        if np.any(valid_mask):
            est_label = f"{est_name}: $\\hat{{d}} = {estimates[est_name]:.2f}$"
            # Plot the line
            ax.plot(d_grid[valid_mask], obj_vals[valid_mask],
                    color=ESTIMATOR_COLORS[est_name],
                    linestyle=ESTIMATOR_LINESTYLES[est_name],
                    linewidth=2,
                    label=est_label,
                    alpha=0.8)

            # Add marker at the minimum
            if not np.isnan(estimates[est_name]):
                # Find the closest point on the grid to the estimate
                closest_idx = np.argmin(np.abs(d_grid - estimates[est_name]))
                if not np.isnan(obj_vals[closest_idx]):
                    ax.plot(estimates[est_name], obj_vals[closest_idx],
                            marker='o',
                            color=ESTIMATOR_COLORS[est_name],
                            markersize=6,
                            markerfacecolor=ESTIMATOR_COLORS[est_name],
                            markeredgewidth=2,
                            markeredgecolor=ESTIMATOR_COLORS[est_name])

    # Formatting
    ax.set_xlabel('$d$', fontsize=12)
    ax.set_ylabel('Objective Function', fontsize=12)
    # ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)


def main():
    plot_objective_functions()


if __name__ == "__main__":
    main()
