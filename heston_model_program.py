
# Title: 
# Author: Ahmet Bajrami, Filippo Simonetti, Thomas Seveso, Devid Vernava
# Place: Lugano, 17.01.2025
# Description: Option Pricing with COS-FFT method applyed to the Heston model



##Credits
"""
Part of this code was adapted from 
[larsphilipp/AdvNum19_COS-FFT](https://github.com/larsphilipp/AdvNum19_COS-FFT/tree/master), "OptionPricing.py" and "AllFunctions.py" files.
    
    - Author   : Elisa FLeissner, Lars Stauffenegger
    - Retrived : December 2024
"""


#---------------------------- library needed ---------------------------------------------------------------------
import yfinance as yf
import numpy as np
import scipy as scp
import scipy.stats as stats
from functools import partial
from scipy.integrate import quad 
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from datetime import datetime, date

#--------------------------- defined functions --------------------------------------------------------------------

def truncationRange(L, mu, tau, sigma, theta, kappa, rho, volvol):
        """
        Truncation Range based on Fitt et al. (2010)
        """
        c1 = mu * tau + (1 - np.exp(-kappa * tau)) * (theta - sigma)/(2 * kappa) - theta * tau / 2
        c2 = 1/(8 * np.power(kappa,3)) * (volvol * tau * kappa * np.exp(-kappa * tau) \
            * (sigma - theta) * (8 * kappa * rho - 4 * volvol) \
            + kappa * rho * volvol * (1 - np.exp(-kappa * tau)) * (16 * theta - 8 * sigma) \
            + 2 * theta * kappa * tau * (-4 * kappa * rho * volvol + np.power(volvol,2) + 4 * np.power(kappa,2)) \
            + np.power(volvol,2) * ((theta - 2 * sigma) * np.exp(-2 * kappa * tau) \
            + theta * (6 * np.exp(-kappa * tau) - 7) + 2 * sigma) \
            + 8 * np.power(kappa,2) * (sigma - theta) * (1 - np.exp(-kappa * tau)))

        a = c1 - L * np.sqrt(np.abs(c2))
        b = c1 + L * np.sqrt(np.abs(c2))
        return a, b


# Cosine series expansion based on Fang & Oosterlee (2008)
    # code borrowed from FLeissner, Stauffenegger(2019). https://github.com/larsphilipp/AdvNum19_COS-FFT/tree/master
def cosSerExp(a, b, c, d, k):
    bma = b-a
    uu  = k * np.pi/bma
    chi = np.multiply(np.divide(1, (1 + np.power(uu,2))), (np.cos(uu * (d-a)) * np.exp(d) - np.cos(uu * (c-a)) * np.exp(c) + np.multiply(uu,np.sin(uu * (d-a))) * np.exp(d)-np.multiply(uu,np.sin(uu * (c-a))) * np.exp(c)))
    return chi


def cosSer1(a, b, c, d, k):
    bma    = b-a
    uu     = k * np.pi/bma
    uu[0]  = 1
    psi    = np.divide(1,uu) * ( np.sin(uu * (d-a)) - np.sin(uu * (c-a)) )
    psi[0] = d-c
    return psi


# definition of objective function for the fitting of parameters
def heston_loss(params, market_prices, strikes, tau, S0, r):
    """
    Objective function for parameters calibration of Heston-Cui model.
    
    params       : array [kappa, theta, sigma, rho, volvol]
    market_prices: array of market prices
    strikes      : array of exercise prices
    tau          : time to expiration
    S0           : initial stock price
    r            : risk-free rate
    """
    kappa, theta, sigma, rho, volvol = params

    # penallization of non valid parameters
    if not (0 < kappa and 0 < theta and 0 < sigma and -1 <= rho <= 1 and 0 < volvol):
        return np.inf
    
    P_COS_CUI = np.zeros((np.size(K)))
    model_prices = np.zeros((np.size(K)))

    # Option prices with the Heston-Cui model
    for m in range(0, np.size(K)):
        x  = np.log(S0/K[m])
        addIntegratedTerm = np.exp(1j * k * np.pi * (x-a)/bma)
        Fk = np.real(characteristicFunctionHCUI * addIntegratedTerm)
        Fk[0]*= 0.5						
        P_COS_CUI[m] = K[m] * np.sum(np.multiply(Fk, UkPut)) * np.exp(-r * tau)
        model_prices[m] = P_COS_CUI[m] + S0 * np.exp(-q * tau) - K[m] * np.exp(-r * tau)

    # residual minimal squared error (RMSE)
    error = sum((np.array(market_prices) - np.array(model_prices))**2)
    return error


#-------------------------- begin code from ChatGPT ---------------------------------------------------

# Function to simulate paths under the Heston model
def simulate_heston_paths(S0, tau, r, kappa, theta, sigma, rho, volvol, N, M):
    """
    Monte Carlo simulation of the Heston model.
    
    S0    : Initial stock price
    tau   : Time to maturity
    r     : Risk-free rate
    kappa : Speed of mean reversion
    theta : Long-term variance level
    sigma : Volatility of volatility
    rho   : Correlation between the Brownian motions
    volvol: Initial variance
    N     : Number of time steps
    M     : Number of paths
    """
    dt = tau / N  # Time step
    S = np.zeros((M, N + 1))
    v = np.zeros((M, N + 1))
    S[:, 0] = S0
    v[:, 0] = volvol

    # Generate correlated Brownian motions
    Z1 = np.random.normal(size=(M, N))
    Z2 = np.random.normal(size=(M, N))
    W1 = Z1
    W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

    for t in range(1, N + 1):
        v[:, t] = np.maximum(v[:, t - 1] + kappa * (theta - v[:, t - 1]) * dt +
                             sigma * np.sqrt(v[:, t - 1] * dt) * W1[:, t - 1], 0)
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * v[:, t - 1]) * dt +
                                       np.sqrt(v[:, t - 1] * dt) * W2[:, t - 1])

    return S

# Function to price options using Monte Carlo simulation
def price_options_mc(S, strikes, tau, r):
    """
    Calculate option prices using Monte Carlo simulation.
    
    S: Simulated paths (M x N+1 array)
    strikes: Array of strike prices
    tau: Time to maturity
    r: Risk-free rate
    """
    S_T = S[:, -1]  # Terminal stock prices
    prices = []
    for K in strikes:
        payoff = np.maximum(S_T - K, 0)             # Call option payoff
        price = np.mean(payoff) * np.exp(-r * tau)  # Discounted payoff
        prices.append(price)
    return np.array(prices)



def plot_simulated_paths(S, tau, N):
    """
    Plot a subset of simulated stock price paths.
    
    S   : Simulated paths (M x N+1 array)
    tau : Time to maturity
    N   : Number of time steps
    """
    time = np.linspace(0, tau, N + 1)
    plt.figure(figsize=(10, 6))
    for i in range(min(10, S.shape[0])):  # Plot the first 10 paths
        plt.plot(time, S[i, :], lw=1)
    plt.title("Simulated Stock Price Paths (Heston Model)")
    plt.xlabel("Time to Maturity")
    plt.ylabel("Stock Price")
    plt.grid()
    plt.show()

# Plot market prices vs Monte Carlo prices
def plot_price_comparison(market_prices, mc_prices, strikes):
    """
    Compare market call option prices with Monte Carlo prices.
    
    params
    market_prices: Array of call market prices
    mc_prices    : Array of call Monte Carlo simulated prices
    strikes      : Array of strike prices

    """
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, market_prices, 'o-', label="Market Prices", lw=2, color="blue")
    plt.plot(strikes, mc_prices, 'x--', label="Monte Carlo Prices", lw=2, color="orange")
    plt.title("Call Market Prices vs Call Monte Carlo Prices")
    plt.xlabel("Strike Prices")
    plt.ylabel("Call Option Price")
    plt.legend()
    plt.grid()
    plt.show()
#--------------------------- end code from ChatGPT -------------------------------------------------------

#-------------------------- Heston characteristic functions ----------------------------------------------

def cf_Heston_original(u, t, v0, mu, kappa, theta, volvol, rho):
    """
    Heston characteristic function as proposed in the original paper of Heston (1993)
    """
    xi = kappa - volvol * rho * u * 1j
    d = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
    g1 = (xi + d) / (xi - d)
    phi = np.exp(
        1j * u * mu * t
        + (kappa * theta) / (sigma**2) * ((xi + d) * t - 2 * np.log((1 - g1 * np.exp(d * t)) / (1 - g1)))
        + (v0 / sigma**2) * (xi + d) * (1 - np.exp(d * t)) / (1 - g1 * np.exp(d * t))
    )
    return phi



def cf_Heston_schoutens(mu, r, u, tau, sigma, theta, kappa, rho, volvol):
    """
    Heston characteristic function based on Schoutens (2004)
    """
    xi = kappa - 1j * rho * volvol * u

    d = np.sqrt(xi**2) + (volvol**2) * ((u**2) + 1j*u)
    g = (xi + d) / (xi - d)
    
    C = np.divide(kappa * theta, np.power(volvol,2)) * ( (xi - d) * tau - 2 * np.log(np.divide((1 - g * np.exp(-d * tau)) , (1-g)) ))
    D = 1j * r * u * tau + sigma / np.power(volvol,2) * (np.divide((1 - np.exp(-d * tau)), (1 - g * np.exp(-d * tau)))) * (xi - d) 
    phi = np.exp(D) * np.exp(C)
    return phi



def cf_Heston_Cui(u, tau, S0, sigma, theta, rho, kappa, volvol , r, q):
    """
    Heston characteristic function as based on Cui et Al. (2019)
    """
    # Parameters for the Cui et al. characteristic function
    F = S0 * np.exp((r - q) * tau)  # Forward price
    xi = kappa - volvol  * rho * 1j * u

    d = np.sqrt(xi**2 + volvol **2 * (u**2 + 1j * u))
    g2 = (xi - d) / (xi + d)
    A1 = ((u**2 + 1j * u) * np.sinh(d * tau)) / (2 * d)
    A2 = (sigma * np.cosh(d * tau) + (xi * sigma * np.sinh(d * tau)) / d)
    A = A1 / A2
    B = (d * np.exp(kappa * tau / 2)) / A2
    D = np.log(d / sigma + (kappa - d) * tau / 2) - np.log((d + xi) / (2 * sigma) + (d - xi) / (2 * sigma) * np.exp(-d * tau))
    
    exponent = 1j * u * np.log(F / S0) - (kappa * theta * rho * tau * 1j * u) / volvol  - A + (2 * kappa * theta / volvol **2) * D
    return np.exp(exponent)


#-------------------------- data ---------------------------------------------------------------------
# Download historical data of a chosen stock with yfinance
ticker = "AAPL"    # modifiable
stock = yf.Ticker(ticker)
stockData = yf.download(ticker, start="2021-01-01" )

# log-return of stocks data and annualized volatility 
logReturn = np.log(stockData['Close']) - np.log(stockData['Close'].shift(1))
logReturn.dropna(inplace=True) 
tradingDaysCount = 252
annualisedMean = np.mean(logReturn) * tradingDaysCount
annualisedVariance = np.var(logReturn) * tradingDaysCount
annualisedStdDev = np.sqrt(annualisedVariance)
lastPrice = stockData["Close"].iloc[-1][0]

   



r          =  0                     # assumption risk-free rate, modifiable
mu         =  r                     # mean rate of drift, modifiable
sigma      =  0.3                   # initial volatility at time 0, modifiable               
S0         =  lastPrice             # today's stock price      
e_date     =  stock.options[8]      # expiration date, modifiable
delta_days =  datetime.strptime(str(e_date), "%Y-%m-%d") - datetime.strptime(str(date.today()), "%Y-%m-%d")  # numbers of days to expiration 
tau        =  delta_days.days/ 365  # time to expiration in years          
q          =  0                     # dividend yield, modifiable
kappa      =  1.5768                # speed of mean revertion, modifiable 
theta      =  0.4                   # mean level of variance, modifiable         
volvol     =  0.6                   # volatility of the volatility, modifiable
rho        = -0.6                   # correlation, modifiable


# truncation range

L       = 80
a, b    = truncationRange(L, mu, tau, sigma, theta, kappa, rho, volvol)
bma     = b-a

# Number of points
N       = 15
k       = np.arange(np.power(2,N))

# Input for the Characterstic Function Phi

u       = k * np.pi/bma

# Option Market prices using stock.options
Options_data  = stock.option_chain(e_date)[0]

market_prices = Options_data["lastPrice"]

K             = Options_data.strike


# Vector of random generated gamma values

gamma_values = stats.norm.rvs(0,1,1000)
print(gamma_values)



# Payoff series coefficients UKPut/UKcall
UkPut  = 2 / bma * ( cosSer1(a,b,a,0,k) - cosSerExp(a,b,a,0,k) )
UkCall = 2 / bma * ( cosSerExp(a,b,0,b,k) - cosSer1(a,b,0,b,k) )


# COS Fang Osterlee(2008) applyed to the Heston-Cui characteristic function
characteristicFunctionHCUI = cf_Heston_Cui(u, tau, S0, sigma, theta, rho, kappa, volvol , r, q)

C_COS_HFO = np.zeros((np.size(K)))
P_COS_CUI = np.zeros((np.size(K)))
C_COS_PCP = np.zeros((np.size(K)))

for m in range(0, np.size(K)):
    x  = np.log(S0/K[m])
    addIntegratedTerm = np.exp(1j * k * np.pi * (x-a)/bma)
    Fk = np.real(characteristicFunctionHCUI * addIntegratedTerm)
    Fk[0]*= 0.5						
    C_COS_HFO[m] = K[m] * np.sum(np.multiply(Fk, UkCall)) * np.exp(-r * tau)
    P_COS_CUI[m] = K[m] * np.sum(np.multiply(Fk, UkPut)) * np.exp(-r * tau)
    C_COS_PCP[m] = P_COS_CUI[m] + S0 * np.exp(-q * tau) - K[m] * np.exp(-r * tau)



print("Fair call option price")
print(C_COS_PCP)




# bounds of the parameters
bounds = [
     (0.01, 5),     # kappa
     (0.01, 1.0),   # theta
     (0.01, 1.0),   # sigma
     (-0.99, 0.99), # rho
     (0.01, 1.0)    # volvol
 ]


initial_guess = [2.0, 0.1, 0.1, 0.0, 0.1] #casual valid parameters

# Data input
market_prices = market_prices   # market option prices
strikes       = K               # Corresponding exercise prices

# optimization with differential Evolution
result = differential_evolution(
    heston_loss, 
    bounds, 
    args=(market_prices, strikes, tau, S0, r),
    strategy ='best1bin', 
    maxiter       = 1000, 
    popsize       = 25, 
    tol           = 0.1**8, 
    mutation      = (0.5, 1),
    recombination = 0.7,
    disp= True
    
)

# results
if result.success:
    print("Optimization completed succesfully!")
    print("Optimal parameters found:", result.x)
else:
    print("Optimization failed.")


#montecarlo simulation
# Monte Carlo parameters
N = 1000  # Number of time steps
M = 10    # Number of paths

# Use optimized parameters from differential_evolution
if result.success:
    kappa, theta, sigma, rho, volvol = result.x
    print("Simulating with parameters:", result.x)

    # Simulate paths under the Heston model
    simulated_paths = simulate_heston_paths(S0, tau, r, kappa, theta, sigma, rho, volvol, N, M)

    # Calculate option prices using Monte Carlo
    mc_prices = price_options_mc(simulated_paths, strikes, tau, r)

    # Compare with market prices
    print("Market Prices:    ", market_prices)
    print("Monte Carlo Prices:", mc_prices)
    print("RMSE:", sum((market_prices - mc_prices)**2))
else:
    print("Optimization failed. Cannot proceed with verification.")




# Monte Carlo parameters


if result.success:
    kappa, theta, sigma, rho, volvol = result.x
    print("Simulating with parameters:", result.x)

    # Simulate paths under the Heston model
    simulated_paths = simulate_heston_paths(S0, tau, r, kappa, theta, sigma, rho, volvol, N, M)

    # Plot simulated paths
    plot_simulated_paths(simulated_paths, tau, N)

    # Calculate option prices using Monte Carlo
    mc_prices = price_options_mc(simulated_paths, strikes, tau, r)

    # Compare with market prices
    print("Market Prices:    ", market_prices)
    print("Monte Carlo Prices:", mc_prices)
    print("SRMSE:", sum((market_prices - mc_prices)**2))

    # Plot comparison of market prices and Monte Carlo prices
    plot_price_comparison(market_prices, mc_prices, strikes)
else:
    print("Optimization failed. Cannot proceed with plotting.")
