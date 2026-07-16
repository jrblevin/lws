#!/bin/bash
#
# Replication script for "Semiparametric Estimation of Fractional
# Integration: An Evaluation of Local Whittle Methods"
#
# This script runs all Monte Carlo simulations and empirical analyses
# in the order they appear in the paper, generating LaTeX tables and
# figures.
#
# Usage: ./replicate.sh
#
# Output:
#
#   - LaTeX table files: tables/mc_*.tex, tables/emp_*.tex
#   - Figure files: figures/mc_*.pdf, figures/emp_*.pdf
#   - Log files: logs/*.log

set -e  # Exit on error

# Ensure logs, tables, and figures directories exist
mkdir -p logs
mkdir -p tables
mkdir -p figures

echo "Replication Script for Semiparametric Estimation of Fractional Integration"
echo "=========================================================================="
echo ""
echo "Started at: $(date)"
echo ""

# Record the system and software environment.  LWLFC uses SciPy's L-BFGS-B,
# whose converged optima can shift slightly across SciPy versions, so the
# exact environment is logged with every replication run.
{
echo "System Information"
echo "------------------"
echo "Host:        $(uname -n)"
echo "Kernel:      $(uname -s) $(uname -r) ($(uname -m))"
case "$(uname -s)" in
    Darwin)
        echo "OS:          $(sw_vers -productName) $(sw_vers -productVersion) (build $(sw_vers -buildVersion))"
        echo "CPU:         $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
        echo "Cores:       $(sysctl -n hw.physicalcpu) physical / $(sysctl -n hw.logicalcpu) logical"
        echo "Memory:      $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
        ;;
    Linux)
        if [ -r /etc/os-release ]; then
            echo "OS:          $(. /etc/os-release && echo "$PRETTY_NAME")"
        fi
        echo "CPU:         $(awk -F': ' '/model name/ {print $2; exit}' /proc/cpuinfo 2>/dev/null || echo unknown)"
        echo "Cores:       $(nproc 2>/dev/null || echo unknown) logical"
        echo "Memory:      $(awk '/MemTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo 2>/dev/null || echo unknown) GB"
        ;;
esac
echo ""
echo "Software Versions"
echo "-----------------"
python3 - <<'PYEOF'
import platform
from importlib.metadata import version, PackageNotFoundError

print(f"{'Python:':<13}{platform.python_version()} ({platform.python_implementation()})")
for pkg in ("pyelw", "numpy", "scipy", "pandas", "matplotlib"):
    try:
        ver = version(pkg)
    except PackageNotFoundError:
        try:
            ver = __import__(pkg).__version__
        except Exception:
            ver = "(not installed)"
    print(f"{pkg + ':':<13}{ver}")
PYEOF
} 2>&1 | tee logs/system_info.log
echo ""

# Monte Carlo Tables and Figures

echo "[1/14] Running Table 1: mc_replications_lw.py (LW, Velasco, and HC Estimators)"
python3 mc_replications_lw.py 2>&1 | tee logs/mc_replications_lw.log
echo ""

echo "[2/14] Running Table 2: mc_replications_elw.py (ELW and 2ELW Estimators)"
python3 mc_replications_elw.py 2>&1 | tee logs/mc_replications_elw.log
echo ""

echo "[3/14] Running Table 15: mc_lw_v_all.py (Comparison of Velasco Tapers)"
python3 mc_lw_v_all.py 2>&1 | tee logs/mc_lw_v_all.log
echo ""

echo "[4/14] Running Tables 3, 10, and 11 (Comprehensive Comparison; n = 500, 250, 1000)"
python3 mc_comprehensive.py 2>&1 | tee logs/mc_comprehensive.log
echo ""

echo "[5/14] Running Table 4: mc_unknown_mean.py (Robustness to Unknown Mean)"
python3 mc_unknown_mean.py 2>&1 | tee logs/mc_unknown_mean.log
echo ""

echo "[6/14] Running Table 5: mc_time_trend.py (Robustness to Time Trend)"
python3 mc_time_trend.py 2>&1 | tee logs/mc_time_trend.log
echo ""

echo "[7/14] Running Figure 1: mc_distributions.py (Sampling Distributions)"
python3 mc_distributions.py 2>&1 | tee logs/mc_distributions.log
echo ""

# Empirical Tables and Figures

echo "[8/14] Running Table 6: emp_hurvich_chen.py (Hurvich and Chen Datasets)"
python3 emp_hurvich_chen.py 2>&1 | tee logs/emp_hurvich_chen.log
echo ""

echo "[9/14] Running Table 7 and Figures 3 and 4: emp_bandwidth_selection.py (S&P 500)"
python3 emp_bandwidth_selection.py 2>&1 | tee logs/emp_bandwidth_selection.log
echo ""

echo "[10/14] Running Table 8 and Figure 5: emp_structural_breaks.py (France CPI)"
python3 emp_structural_breaks.py 2>&1 | tee logs/emp_structural_breaks.log
echo ""

echo "[11/14] Running Figure 2: emp_objective.py (Objective Functions)"
python3 emp_objective.py 2>&1 | tee logs/emp_objective.log
echo ""

# Appendix: Robustness Checks

echo "[12/14] Running Table 12: mc_comprehensive_ma.py (MA(1) Short-Run Dynamics)"
python3 mc_comprehensive_ma.py 2>&1 | tee logs/mc_comprehensive_ma.log
echo ""

echo "[13/14] Running Table 13: mc_comprehensive_arma.py (ARMA(1,1) Short-Run Dynamics)"
python3 mc_comprehensive_arma.py 2>&1 | tee logs/mc_comprehensive_arma.log
echo ""

echo "[14/14] Running Table 14: mc_comprehensive_heavy.py (Heavy-Tailed and GARCH Innovations)"
python3 mc_comprehensive_heavy.py 2>&1 | tee logs/mc_comprehensive_heavy.log
echo ""

echo "Replication Complete!"
echo ""
echo "Monte Carlo Tables (tables/):"
echo "  - mc_replications_lw.tex"
echo "  - mc_replications_elw.tex"
echo "  - mc_comprehensive.tex"
echo "  - mc_unknown_mean.tex"
echo "  - mc_time_trend.tex"
echo ""
echo "Empirical Tables (tables/):"
echo "  - emp_hurvich_chen.tex"
echo "  - emp_bandwidth_selection.tex"
echo "  - emp_structural_breaks.tex"
echo ""
echo "Figures (figures/):"
echo "  - mc_distributions.pdf"
echo "  - emp_objective.pdf"
echo "  - emp_bandwidth_selection.pdf"
echo "  - emp_bandwidth_mse.pdf"
echo "  - emp_structural_breaks.pdf"
echo ""
echo "Appendix Tables (tables/):"
echo "  - mc_comprehensive_n250.tex"
echo "  - mc_comprehensive_n1000.tex"
echo "  - mc_comprehensive_ma1.tex"
echo "  - mc_comprehensive_arma1.tex"
echo "  - mc_comprehensive_heavy.tex"
echo "  - mc_lw_v_all.tex"
echo ""
echo "Logs saved to: logs/"
echo ""
echo "Completed at: $(date)"
