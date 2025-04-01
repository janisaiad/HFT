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
from scipy.optimize import minimize
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
        
        # Create plots
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Hill estimator vs k
        plt.subplot(1, 2, 1)
        plt.plot(k_values, hill_estimates, 'b-', label='Estimateur de Hill')
        plt.axhline(y=np.mean(hill_estimates), color='r', linestyle='--',
                   label=f'Moyenne α ≈ {np.mean(hill_estimates):.2f}')
        plt.xlabel('k')
        plt.ylabel('Estimateur de Hill α')
        plt.title(f'Estimateur de Hill vs k - {process_name}')
        plt.legend()
        
        # Plot 2: Inter-arrival time distribution
        plt.subplot(1, 2, 2)
        plt.hist(inter_arrival_times, bins=50, density=True, alpha=0.7)
        plt.xlabel('Temps inter-arrivée (s)')
        plt.ylabel('Densité')
        plt.title(f'Distribution des temps inter-arrivée - {process_name}')
        
        plt.tight_layout()
        plt.savefig(f'inter_arrival_times_analysis_{process_name}.png')
        plt.close()
    else:
        print(f"Pas assez d'événements pour analyser {process_name}")
