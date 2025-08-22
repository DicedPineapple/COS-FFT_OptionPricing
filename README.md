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

## Configuration
Users can customize the following parameters in the code:
- **Stock selection**
  - **ticker**: Stock symbol (default: "AAPL") - Change to any valid stock symbol like "MSFT", "GOOGL", etc.
- **Model parameters**
  - **r**: Risk-free rate (default: 0) - Set to appropriate current risk-free rate
  - **sigma**: Initial volatility at time 0 (default: 0.3) - Adjust based on market conditions
  - **kappa**: Speed of mean reversion (default: 1.5768) - Higher values mean faster reversion to mean volatility
  - **theta**: Long-term variance level (default: 0.4) - The equilibrium volatility level
  - **volvol**: Volatility of volatility (default: 0.6) - Controls how much volatility fluctuates
  - **rho**: Correlation between stock price and volatility (default: -0.6) - Typically negative for equity markets

- **Expiration settings**
  - **e_date**: Expiration date selection (default: stock.options[8]) - Choose from available expiration dates by changing the index

