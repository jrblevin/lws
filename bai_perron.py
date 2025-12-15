"""
Bai and Perron (1998, 2003) structural break detection

Implements the Bai-Perron dynamic programming algorithm for finding
globally optimal break locations and BIC-based model selection.

Note: This is a partial implementation focused on break detection and
model selection. It does not include the supF test statistics or
associated critical values from Bai and Perron (1998, 2003a).

References
----------

- Bai, J. and P. Perron (1998). Estimating and testing linear models
  with multiple structural changes. _Econometrica_ 66, 47--78.

- Bai, J. and P. Perron (2003a). Critical values for multiple
  structural change tests. _Econometrics Journal_ 6, 72--78.

- Bai, J. and P. Perron (2003b). Computation and analysis of multiple
  structural change models. _Journal of Applied Econometrics_ 18, 1--22.
"""

import numpy as np


def ols_ssr(y, X):
    """
    Compute OLS sum of squared residuals.

    Parameters
    ----------
    y : array-like
        Dependent variable
    X : array-like
        Independent variables (including constant)

    Returns
    -------
    float
        Sum of squared residuals
    """
    y = np.asarray(y)
    X = np.asarray(X)

    if len(y) < X.shape[1]:
        return np.inf

    # OLS: beta = (X'X)^{-1} X'y
    # SSR = y'y - y'X (X'X)^{-1} X'y
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        return np.sum(residuals ** 2)
    except np.linalg.LinAlgError:
        return np.inf


def compute_segment_ssr(endog, exog, start, end):
    """
    Compute sum of squared residuals for a segment.

    Parameters
    ----------
    endog : array-like
        Dependent variable
    exog : array-like
        Independent variables (including constant)
    start : int
        Start index (inclusive)
    end : int
        End index (exclusive)

    Returns
    -------
    float
        Sum of squared residuals
    """
    if end - start < exog.shape[1]:
        return np.inf
    return ols_ssr(endog[start:end], exog[start:end])


def build_ssr_matrix(endog, exog, trim):
    """
    Build upper triangular matrix of SSR values for all possible segments.

    This is the key computational step - we compute SSR for every possible
    segment [i, j] where j - i >= trim.

    Parameters
    ----------
    endog : array-like
        Dependent variable
    exog : array-like
        Independent variables
    trim : int
        Minimum segment size

    Returns
    -------
    np.ndarray
        Upper triangular matrix where entry [i, j] contains SSR for
        segment from i to j+1 (i.e., observations i through j inclusive)
    """
    n = len(endog)
    ssr_matrix = np.full((n, n), np.inf)

    for i in range(n):
        for j in range(i + trim - 1, n):
            # Segment from i to j+1 (j inclusive)
            ssr_matrix[i, j] = compute_segment_ssr(endog, exog, i, j + 1)

    return ssr_matrix


def find_breakpoints(endog, exog=None, nbreaks=1, trim=0.15):
    """
    Find optimal breakpoint locations using dynamic programming.

    Uses the Bai and Perron (2003b) algorithm to find globally optimal
    break locations that minimize total sum of squared residuals.

    Parameters
    ----------
    endog : array-like
        The dependent variable (e.g., inflation series)
    exog : array-like, optional
        The independent variables. If None, uses a constant (mean-shift model)
    nbreaks : int
        Number of breakpoints to find
    trim : float or int
        Minimum segment size. If float < 1, interpreted as fraction of n.

    Returns
    -------
    breakpoints : tuple
        Zero-indexed breakpoint locations. The k-th breakpoint is the last
        observation in regime k (so regime k+1 starts at breakpoint[k]+1)
    ssr : float
        Total sum of squared residuals with optimal breaks
    """
    endog = np.asarray(endog)
    n = len(endog)

    # Handle exog
    if exog is None:
        # Mean-shift model: just a constant
        exog = np.ones((n, 1))
    else:
        exog = np.asarray(exog)
        if exog.ndim == 1:
            exog = exog[:, None]

    k = exog.shape[1]  # Number of regressors

    # Convert trim to integer
    if trim < 1:
        trim = int(np.floor(n * trim))
    trim = max(trim, k + 1)  # Need at least k+1 observations per segment

    # Check feasibility
    if n < (nbreaks + 1) * trim:
        raise ValueError(f"Not enough observations for {nbreaks} breaks "
                         f"with trim={trim}. Need n >= {(nbreaks + 1) * trim}")

    # Build SSR matrix for all possible segments
    ssr_matrix = build_ssr_matrix(endog, exog, trim)

    if nbreaks == 0:
        return (), ssr_matrix[0, n - 1]

    # Dynamic programming to find optimal breaks
    # optimal[m][j] = (min_ssr, breakpoints) for m breaks ending at j
    # where breakpoints is a tuple of break locations

    # Initialize for m=1 break
    optimal = [{}]
    for j in range(2 * trim - 1, n):
        # One break at position bp means:
        # - Segment 1: [0, bp]
        # - Segment 2: [bp+1, j]
        min_ssr = np.inf
        best_bp = None
        for bp in range(trim - 1, j - trim + 1):
            ssr = ssr_matrix[0, bp] + ssr_matrix[bp + 1, j]
            if ssr < min_ssr:
                min_ssr = ssr
                best_bp = bp
        if best_bp is not None:
            optimal[0][j] = (min_ssr, (best_bp,))

    # For m > 1 breaks
    for m in range(2, nbreaks + 1):
        optimal.append({})
        for j in range((m + 1) * trim - 1, n):
            min_ssr = np.inf
            best_bps = None
            # Last break at position bp
            for bp in range(m * trim - 1, j - trim + 1):
                if bp not in optimal[m - 2]:
                    continue
                prev_ssr, prev_bps = optimal[m - 2][bp]
                ssr = prev_ssr + ssr_matrix[bp + 1, j]
                if ssr < min_ssr:
                    min_ssr = ssr
                    best_bps = prev_bps + (bp,)
            if best_bps is not None:
                optimal[m - 1][j] = (min_ssr, best_bps)

    # Extract solution
    if n - 1 not in optimal[nbreaks - 1]:
        raise ValueError("Could not find valid breakpoints")

    min_ssr, breakpoints = optimal[nbreaks - 1][n - 1]
    return breakpoints, min_ssr


def compute_bic(endog, exog, breakpoints):
    """
    Compute BIC for a model with given breakpoints.

    Parameters
    ----------
    endog : array-like
        Dependent variable
    exog : array-like or None
        Independent variables (None for mean-shift model)
    breakpoints : tuple
        Break locations

    Returns
    -------
    float
        BIC value (lower is better)
    """
    endog = np.asarray(endog)
    n = len(endog)

    if exog is None:
        exog = np.ones((n, 1))
    else:
        exog = np.asarray(exog)
        if exog.ndim == 1:
            exog = exog[:, None]

    k = exog.shape[1]
    nbreaks = len(breakpoints)

    # Compute total SSR
    segments = [0] + [bp + 1 for bp in breakpoints] + [n]
    total_ssr = 0
    for i in range(len(segments) - 1):
        start, end = segments[i], segments[i + 1]
        total_ssr += ols_ssr(endog[start:end], exog[start:end])

    # Number of parameters: k parameters per regime + break locations
    # Following Bai and Perron (2003b), the penalty includes both the
    # regression parameters (k per regime) and the break locations
    n_params = k * (nbreaks + 1) + nbreaks

    # BIC = n * log(SSR/n) + n_params * log(n)
    bic = n * np.log(total_ssr / n) + n_params * np.log(n)

    return bic


def select_breaks_bic(endog, exog=None, max_breaks=5, trim=0.15):
    """
    Select optimal number of breaks using BIC.

    Parameters
    ----------
    endog : array-like
        Dependent variable
    exog : array-like or None
        Independent variables
    max_breaks : int
        Maximum number of breaks to consider
    trim : float or int
        Minimum segment size

    Returns
    -------
    optimal_nbreaks : int
        BIC-optimal number of breaks
    all_results : list of dict
        Results for each number of breaks
    """
    endog = np.asarray(endog)
    n = len(endog)

    if exog is None:
        exog = np.ones((n, 1))

    results = []

    for nbreaks in range(max_breaks + 1):
        try:
            if nbreaks == 0:
                breakpoints = ()
                ssr = ols_ssr(endog, exog)
            else:
                breakpoints, ssr = find_breakpoints(endog, exog, nbreaks, trim)

            bic = compute_bic(endog, exog, breakpoints)
            results.append({
                'nbreaks': nbreaks,
                'breakpoints': breakpoints,
                'ssr': ssr,
                'bic': bic
            })
        except ValueError as e:
            # Not enough observations for this many breaks
            break

    optimal_idx = np.argmin([r['bic'] for r in results])
    optimal_nbreaks = results[optimal_idx]['nbreaks']

    return optimal_nbreaks, results
