# Heston Model Option Pricing with COS-FFT Method

A Python implementation of option pricing using the COS-FFT method applied to the Heston stochastic volatility model.

## Project Overview

This project implements the COS-FFT (Fourier Cosine Series Expansion with Fast Fourier Transform) method for pricing European options under the Heston stochastic volatility model. The implementation includes parameter calibration using market data and Monte Carlo simulation for verification.

## Features

- **Heston Model Implementation**: Three different characteristic function implementations:
  - Original Heston (1993) formulation
  - Schoutens (2004) formulation
  - Cui et al. (2019) formulation
  
- **COS-FFT Method**: Efficient option pricing using Fourier cosine series expansion

- **Parameter Calibration**: Differential evolution optimization to fit model parameters to market data

- **Monte Carlo Simulation**: Path simulation under the Heston model for verification

- **Market Data Integration**: Automatic download of real option chain data using Yahoo Finance
