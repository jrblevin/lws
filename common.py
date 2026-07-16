"""
Common utilities for replication scripts.
"""

import numpy as np
from pyelw.simulate import fracdiff

# Threshold for highlighting large MSE values in LaTeX tables
MSE_THRESHOLD = 0.05

# Threshold for highlighting large MSE ratio values in LaTeX tables
MSE_RATIO_THRESHOLD = 2.0

# Common color scheme for estimators across all plots
# Uses ColorBrewer qualitative colors for clarity and accessibility
ESTIMATOR_COLORS = {
    'LW': '#1f77b4',       # Blue - Local Whittle
    'Velasco': '#ff7f0e',  # Orange - Velasco (Kolmogorov) taper
    'V': '#ff7f0e',        # Orange - Velasco (short name)
    'HC': '#2ca02c',       # Green - Hurvich-Chen taper
    'ELW': '#d62728',      # Red - Exact Local Whittle
    'ELW-DM': '#8c564b',   # Brown - Exact Local Whittle (demeaned data)
    '2ELW': '#9467bd',     # Purple - Two-step ELW
}

# Line styles for estimators (if needed for differentiation)
ESTIMATOR_LINESTYLES = {
    'LW': '-',
    'Velasco': '-',
    'V': '-',
    'HC': '-',
    'ELW': '-',
    'ELW-DM': '-',
    '2ELW': '-',
}

def format_mse_latex(mse_val, threshold=MSE_THRESHOLD, decimals=3, width=7):
    """Wrap MSE value in \\largemse{} if it exceeds threshold."""
    if mse_val != mse_val:  # NaN check
        return "--" if width < 7 else "---"

    formatted = f"{mse_val:{width}.{decimals}f}"

    if mse_val > threshold:
        return f"\\largemse{{{formatted}}}"
    return formatted


def format_mse_ratio_latex(ratio_val, threshold=MSE_RATIO_THRESHOLD, decimals=3, width=7):
    """Wrap MSE ratio value in \\largemse{} if it exceeds threshold."""
    if ratio_val != ratio_val:  # NaN check
        return "--"

    formatted = f"{ratio_val:{width}.{decimals}f}"

    if ratio_val > threshold:
        return f"\\largemse{{{formatted}}}"
    return formatted


def arfima_arma(n, d, phi=0.0, theta=0.0, sigma=1.0, seed=None, burnin=0):
    r"""
    Simulate an ARFIMA(1,d,1) process:

        (1 - \phi L)(1-L)^d X_t = (1 + \theta L) \epsilon_t.

    Generalizes pyelw.simulate.arfima (which handles ARFIMA(1,d,0)) to allow an
    MA(1) short-run component, used for the short-run-specification robustness
    check (Appendix A). It nests both special cases bit-for-bit, given the same
    seed: theta=0 reproduces arfima(n, d, phi), and phi=0 reproduces an
    ARFIMA(0,d,1) (MA-only) process. The AR(1) recursion uses the exact
    ARMA(1,1) stationary initialization, which collapses to arfima's AR(1)
    initialization when theta=0.

    Algorithm:
    1. Generate MA(1) errors: v_t = \epsilon_t + \theta \epsilon_{t-1}.
    2. Apply the AR(1) recursion: u_t = \phi u_{t-1} + v_t.
    3. Apply the fractional filter: X_t = (1-L)^{-d} u_t.
    4. Discard burn-in observations.

    Parameters
    ----------
    n : int
        Sample size (final output length).
    d : float
        Fractional differencing parameter.
    phi : float, default=0.0
        AR(1) coefficient.
    theta : float, default=0.0
        MA(1) coefficient.
    sigma : float, default=1.0
        Innovation standard deviation.
    seed : int, optional
        Random seed for reproducibility.
    burnin : int, default=0
        Number of burn-in observations to discard.

    Returns
    -------
    np.ndarray
        ARFIMA(1,d,1) process of length n.
    """
    if seed is not None:
        np.random.seed(seed)

    n_total = n + burnin

    # Step 1: MA(1) errors, v_t = eps_t + theta * eps_{t-1} (eps_{-1} = 0).
    eps = np.random.normal(0, sigma, n_total)
    v = eps.copy()
    if abs(theta) >= 1e-13:
        v[1:] += theta * eps[:-1]

    # Step 2: AR(1) recursion, u_t = phi * u_{t-1} + v_t.
    if abs(phi) < 1e-13:
        u = v
    else:
        u = np.zeros(n_total)
        if abs(phi) < 1:
            # Exact ARMA(1,1) stationary standard deviation, factored so that the
            # theta = 0 case multiplies by exactly 1.0 and is bit-identical to
            # arfima's AR(1) initialization sigma / sqrt(1 - phi^2).
            std0 = (sigma / np.sqrt(1.0 - phi**2)
                    * np.sqrt(1.0 + 2.0 * phi * theta + theta**2))
            u[0] = np.random.normal(0, std0)
        else:
            u[0] = v[0]
        for t in range(1, n_total):
            u[t] = phi * u[t-1] + v[t]

    # Step 3: Apply fractional filter if needed.
    if abs(d) < 1e-13:
        return u[burnin:]

    x = fracdiff(u, -d)

    # Step 4: Discard burn-in.
    return x[burnin:]


def write_replication_table(filename, caption, label, panels, notes):
    """
    Write a combined Original-vs-Replication LaTeX table.

    Parameters
    ----------
    filename : str
        Output path for the .tex file.
    caption : str
        Table caption.
    label : str
        LaTeX cross-reference label.
    panels : list of (str, list of tuple)
        (title, rows) pairs, one per panel, where each row is a
        (d, orig_bias, orig_sd, orig_mse, rep_bias, rep_sd, rep_mse)
        tuple.
    notes : str
        Table notes (LaTeX source).
    """
    latex_table = f"""\\begin{{table}}[!tp]
\\centering
\\singlespacing
\\begin{{threeparttable}}
\\caption{{{caption}}}
\\label{{{label}}}
\\footnotesize
\\begin{{tabular}}{{r@{{\\hspace{{1.5em}}}}rrr@{{\\hspace{{1.5em}}}}rrr}}
\\toprule
\\multicolumn{{1}}{{c}}{{}} & \\multicolumn{{3}}{{c}}{{Original}} & \\multicolumn{{3}}{{c}}{{Replication}} \\\\
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}}
$d$ & Bias & S.D. & MSE & Bias & S.D. & MSE \\\\
"""

    for title, rows in panels:
        latex_table += f"""\\midrule
\\multicolumn{{7}}{{c}}{{\\textit{{{title}}}}} \\\\
\\midrule
"""
        for d, *stats in rows:
            cells = ' & '.join(f'${float(v):7.4f}$' for v in stats)
            latex_table += f"${d:4.1f}$ & {cells} \\\\\n"

    latex_table += f"""\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\footnotesize
\\item Notes: {notes}
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
"""

    with open(filename, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {filename}")
