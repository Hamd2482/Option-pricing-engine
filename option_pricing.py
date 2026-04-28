import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

# ================= BLACK-SCHOLES =================
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

# ================= MONTE CARLO =================
def monte_carlo_call(S, K, T, r, sigma, simulations=10000):
    Z = np.random.randn(simulations)
    ST = S * np.exp((r - 0.5 * sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r*T) * np.mean(payoff)

# ================= GREEKS =================
def delta_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def gamma_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

# ================= USER INPUT =================
print("=== Option Pricing Engine ===")

S = float(input("Enter Stock Price (S): "))
K = float(input("Enter Strike Price (K): "))
T = float(input("Enter Time to Maturity (T in years): "))
r = float(input("Enter Interest Rate (r): "))
sigma = float(input("Enter Volatility (sigma): "))

# ================= PRICING =================
bs_price = black_scholes_call(S, K, T, r, sigma)
mc_price = monte_carlo_call(S, K, T, r, sigma, 100000)

print("\n===== PRICING =====")
print(f"Black-Scholes Price: {bs_price:.4f}")
print(f"Monte Carlo Price: {mc_price:.4f}")

# ================= GREEKS =================
delta = delta_call(S, K, T, r, sigma)
gamma = gamma_call(S, K, T, r, sigma)

print("\n===== GREEKS =====")
print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.6f}")

# ================= EXPERIMENTATION =================
print("\n===== VOLATILITY EXPERIMENT =====")
for vol in [0.1, 0.2, 0.3]:
    price = black_scholes_call(S, K, T, r, vol)
    print(f"Volatility {vol}: Price = {price:.4f}")

# ================= CONVERGENCE =================
simulations_list = [100, 500, 1000, 5000, 10000, 50000]
mc_results = []

for sims in simulations_list:
    price = monte_carlo_call(S, K, T, r, sigma, sims)
    mc_results.append(price)

# ================= ERROR =================
errors = [abs(p - bs_price) for p in mc_results]

# ================= SAVE CSV =================
df = pd.DataFrame({
    "simulations": simulations_list,
    "mc_price": mc_results,
    "error": errors
})

df.to_csv("results.csv", index=False)

print("\nResults saved to results.csv")

# ================= PLOT =================
plt.figure(figsize=(8,5))
plt.plot(simulations_list, mc_results, marker='o', label="Monte Carlo")
plt.axhline(bs_price, color='r', linestyle='--', label="Black-Scholes")

plt.xscale('log')
plt.xlabel("Simulations (log scale)")
plt.ylabel("Option Price")
plt.title("Monte Carlo Convergence")
plt.legend()
plt.grid()

plt.show()

# ================= ERROR PLOT =================
plt.figure(figsize=(8,5))
plt.plot(simulations_list, errors, marker='o')
plt.xscale('log')
plt.xlabel("Simulations (log scale)")
plt.ylabel("Error")
plt.title("Error Convergence")
plt.grid()

plt.show()