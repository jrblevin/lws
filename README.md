# Replication Code

This directory contains replication code for the following paper:

Blevins, J.R. (2025).
[Semiparametric Estimation of Fractional Integration: An Evaluation of Local Whittle Methods](https://jblevins.org/research/lws).
Working Paper, The Ohio State University.

## Requirements

- Python 3.8+
- NumPy, SciPy, Pandas, Matplotlib
- [PyELW](https://jblevins.org/research/pyelw) package

## Usage

To run all simulations and generate all tables and figures in the
paper:

```bash
./replicate.sh
```

## Files

**Monte Carlo Simulations:**

- `mc_lw.py` - Local Whittle estimator
- `mc_lw_v.py` - Velasco (1999) tapered LW estimator
- `mc_lw_v_all.py` - Comparison of Velasco (1999) tapers
- `mc_lw_hc.py` - Hurvich and Chen (2000) tapered LW estimator
- `mc_elw.py` - Exact Local Whittle estimator
- `mc_2elw.py` - Two-step ELW estimator
- `mc_comprehensive.py` - Comprehensive comparison
- `mc_unknown_mean.py` - Robustness to unknown mean
- `mc_time_trend.py` - Robustness to time trend
- `mc_distributions.py` - Sampling distributions figure

**Empirical Analyses:**
- `emp_hurvich_chen.py` - Hurvich and Chen (2000) datasets
- `emp_bandwidth_selection.py` - Bandwidth selection (S&P 500)
- `emp_structural_breaks.py` - Structural breaks (France CPI)
- `emp_objective.py` - Objective functions figure

**Supporting Files:**
- `common.py` - Shared utilities
- `bai_perron.py` - Bai-Perron break detection
- `qu_test.py` - Qu (2011) spurious long memory test

## Output

- `tables/` - LaTeX tables
- `figures/` - PDF figures
- `logs/` - Log files
