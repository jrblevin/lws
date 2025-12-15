"""
Common utilities for replication scripts.
"""

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
