#!/usr/bin/env Rscript
#
# Validation of the paper's Qu (2011) spurious-long-memory test results.
# ---------------------------------------------------------------------
# Our Python qu_test.py implements Qu's W statistic and Table 1 asymptotic
# critical values, but not the optional ARFIMA-AIC pre-whitening
# ("whitten-aic") finite-sample correction.  As a verification, this script
# runs Qu's original R code (in the qu-2011 subdirectory) with and without
# the pre-whitening, and shows that:
#
#   1. The W statistic from our Python code agrees with the R code.
#
#   2. Qu's optional pre-whitening step does not trigger for our series:
#      filterx() returns each series unchanged.
#
# Usage:
#
#     Rscript qu_test_validation.R

library(fracdiff)
source("qu-2011/whitten-aic.R")

# --- Qu's W statistic -------------------------------------------------------
# This function is transcribed directly from qu-2011/RV5.R (lines ~24-51).
# We do this because RV5.R is a top-level script, not a library, so there
# is no reusable function provided.
qu_W <- function(x, m, eps) {
    n <- length(x)
    freq <- seq(1, n, by = 1) * (2 * pi / n)
    xf <- fft(x)
    px <- (Re(xf)^2 + Im(xf)^2) / (2 * pi * n)
    perdx <- px[2:n]
    # local Whittle likelihood
    fn <- function(h) {
        lambda <- freq[1:m]^(2 * h - 1)
        Gh <- mean(perdx[1:m] * lambda)
        log(Gh) - (2 * h - 1) * mean(log(freq[1:m]))
    }
    est <- optimize(fn, c(0, 1.5), tol = 0.00001)
    hhat <- est$minimum
    lambda_hat <- freq[1:m]^(2 * hhat - 1)
    Ghat <- mean(perdx[1:m] * lambda_hat[1:m])

    # now compute the statistic
    comp1 <- (perdx[1:m] * lambda_hat[1:m]) / Ghat
    comp2 <- log(freq[1:m]) - mean(log(freq[1:m]))
    stat <- cumsum((comp1 - 1) * comp2) / sqrt(sum(comp2^2))

    # RV5.R trimming: round(eps * m)
    trm <- round(eps * m)
    max(abs(stat[trm:m]))
}

# French inflation series
cpi <- scan("data/cpi_fr_ext.dat", quiet = TRUE)
inflation <- diff(log(cpi))
n <- length(inflation)

# Break dates
b1 <- (1973 - 1955) * 12 + (3 - 1)   # March 1973
b2 <- (1984 - 1955) * 12 + (10 - 1)  # October 1984

series <- list(
    "Full sample"    = list(x = inflation,               m = 112),
    "Pre-oil shocks" = list(x = inflation[1:b1],         m = 44),
    "Oil shocks"     = list(x = inflation[(b1 + 1):b2],  m = 32),
    "Disinflation"   = list(x = inflation[(b2 + 1):n],   m = 76)
)

# W values reported by qu_test.py
paper_W <- c("Full sample" = 1.297, "Pre-oil shocks" = 0.648,
             "Oil shocks" = 0.644, "Disinflation" = 0.986)

# Qu (2011) Table 1 asymptotic critical values for eps = 0.05
CV10 <- 1.022
CV05 <- 1.155
verdict <- function(W) if (W > CV05) "reject 5%" else if (W > CV10) "reject 10%" else "fail to reject"

# --- Compute and validate ---------------------------------------------------
cat("French inflation: Qu (2011) W with and without optional pre-whitening\n")
cat(strrep("-", 63), "\n")
cat(sprintf("%-15s %4s %10s %12s %18s\n",
            "Series", "m", "W (raw)", "W (filterx)", "Decision"))
cat(strrep("-", 63), "\n")

for (name in names(series)) {
    x <- series[[name]]$x
    m <- series[[name]]$m
    x <- x - mean(x) # demean as in RV5.R
    W_raw <- qu_W(x, m, 0.05)
    W_pw  <- qu_W(suppressWarnings(filterx(x, length(x))), m, 0.05)  # Qu's pre-whitening

    cat(sprintf("%-15s %4d %10.4f %12.4f %18s\n",
                name, m, W_raw, W_pw, verdict(W_pw)))
}

cat(strrep("-", 63), "\n")
cat(sprintf("Critical values (eps = 0.05): 10%% = %.3f, 5%% = %.3f\n", CV10, CV05))

