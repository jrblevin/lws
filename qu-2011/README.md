# Original R Code for Qu (2011) Test

Zhongjun Qu's original R implementation of the test against spurious long
memory, retrieved from <https://sites.bu.edu/qu/codes/> on July 19, 2026.

Original description of the package:

> R code for implementing the test in A Test against Spurious Long Memory
> (Journal of Business and Economic Statistics, 2011). The package contains a
> main file, an optional pre-whitening procedure (whitten-aic), and a data set
> for illustration.

## Contents

- `RV5.R` — main file: computes the W statistic on the series.
- `whitten-aic.R` — optional ARFIMA-AIC pre-whitening procedure (`filterx`).
- `RV5min.txt` — illustrative dataset (5-minute realized volatility).

## Reference

Qu, Z. (2011). "A Test Against Spurious Long Memory." _Journal of Business &
Economic Statistics_ 29(3), 423–438.
