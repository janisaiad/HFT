# ---
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

import os
import polars as pl
import dotenv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
from scipy.stats import norm
from tqdm import tqdm

dotenv.load_dotenv()

def hill_estimator(data, k):
    """Compute Hill estimator for tail index"""
    sorted_data = np.sort(data)[::-1]  # Sort in descending order
    if k >= len(sorted_data):
        return None
    log_ratios = np.log(sorted_data[:k] / sorted_data[k])
    return k / np.sum(log_ratios)

def power_law_kernel(t, alpha, beta, theta):
    """Power law kernel for Hawkes process with flexible exponent"""
    return alpha * np.power(1 + beta*t, -theta)

def fit_hawkes_kernel(event_times, T_max=None):
    """Fit Hawkes kernel parameters using MLE"""
    if T_max is None:
        T_max = event_times[-1]
        
    def neg_log_likelihood(params):
        alpha, beta, theta = params
        intensity = np.zeros_like(event_times)
        for i, t in enumerate(event_times):
            past_events = event_times[event_times < t]
            if len(past_events) > 0:
                intensity[i] = np.sum(power_law_kernel(t - past_events, alpha, beta, theta))
        return -np.sum(np.log(intensity + 1e-10)) + alpha/beta * (T_max - event_times[0])
    
    result = minimize(neg_log_likelihood, x0=[0.1, 1.0, 1.5], 
                     bounds=[(0.01, 10), (0.01, 10), (0.1, 5)])
    return result.x

def fit_power_law(x, y):
    """Fit power law to data in log-log space"""
    log_x = np.log(x[y>0])
    log_y = np.log(y[y>0])
    
    def linear_fit(x, a, b):
        return a + b*x
    
    popt, _ = curve_fit(linear_fit, log_x, log_y)
    return np.exp(popt[0]), popt[1]  # A and beta where y = A*x^beta

# Load GOOGL data
df = pl.read_parquet("/home/janis/HFTP2/HFT/data/DB_MBP_10/data/hawkes_dataset/GOOGL/GOOGL_2024-08-08.parquet")

# Define counting processes based on the columns
counting_processes = {
    "P_a": df["P_a"],
    "P_b": df["P_b"],
    "T_a": df["T_a"],
    "T_b": df["T_b"], 
    "L_a": df["L_a"],
    "L_b": df["L_b"],
    "C_a": df["C_a"],
    "C_b": df["C_b"]
}

for process_name, events in counting_processes.items():
    print(f"\nAnalyse du processus {process_name}")
    
    # Convert to inter-arrival times
    events_np = df["ts_event"].to_numpy()
    counts = events.to_numpy()
    
    if len(events_np) > 1:
        start_time = events_np[0]
        event_times_sec = (events_np - start_time).astype('timedelta64[ns]').astype(np.float64) / 1e9
        
        # Get times where count changes
        event_indices = np.where(np.diff(counts) != 0)[0] + 1
        event_times = event_times_sec[event_indices]
        inter_arrival_times = np.diff(event_times)
        
        # Calculate Hill estimator
        k_values = np.arange(10, min(100, len(inter_arrival_times)))
        hill_estimates = [hill_estimator(inter_arrival_times, k) for k in k_values]
        
        print(f"Nombre d'événements: {len(event_indices)}")
        print(f"Moyenne estimateur de Hill: {np.mean(hill_estimates):.4f}")
        
        # Fit Hawkes process
        alpha, beta, theta = fit_hawkes_kernel(event_times)
        print(f"Paramètres du noyau Hawkes: alpha={alpha:.4f}, beta={beta:.4f}, theta={theta:.4f}")
        
        # Create plots
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Hill estimator vs k
        plt.subplot(1, 3, 1)
        plt.plot(k_values, hill_estimates, 'b-', label='Estimateur de Hill')
        plt.axhline(y=np.mean(hill_estimates), color='r', linestyle='--',
                   label=f'Moyenne α ≈ {np.mean(hill_estimates):.2f}')
        plt.xlabel('k')
        plt.ylabel('Estimateur de Hill α')
        plt.title(f'Estimateur de Hill vs k - {process_name}')
        plt.legend()
        
        # Plot 2: Inter-arrival time distribution
        plt.subplot(1, 3, 2)
        plt.hist(inter_arrival_times, bins=50, density=True, alpha=0.7)
        plt.xlabel('Temps inter-arrivée (s)')
        plt.ylabel('Densité')
        plt.title(f'Distribution des temps inter-arrivée - {process_name}')
        
        # Plot 3: Fitted Hawkes kernel and power law fit
        plt.subplot(1, 3, 3)
        t_range = np.logspace(-3, 1, 1000)
        kernel_values = power_law_kernel(t_range, alpha, beta, theta)
        
        # Fit power law to kernel values
        A, beta_power = fit_power_law(t_range, kernel_values)
        power_law_fit = A * np.power(t_range, beta_power)
        
        plt.plot(t_range, kernel_values, 'r-', label='Noyau Hawkes')
        plt.plot(t_range, power_law_fit, 'b--', 
                label=f'Loi de puissance: t^({beta_power:.2f})')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Temps (s)')
        plt.ylabel('Intensité')
        plt.title(f'Noyau Hawkes et fit en loi de puissance\nθ={theta:.2f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'hawkes_analysis_{process_name}.png')
        plt.close()
    else:
        print(f"Pas assez d'événements pour analyser {process_name}")
