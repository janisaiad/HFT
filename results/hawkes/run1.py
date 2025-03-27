import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import polars as pl
from hft.models.hawkes import Hawkes
import time
from datetime import datetime, timedelta
import dotenv

dotenv.load_dotenv()


# Set up parameters
N = 8  # 8 event types
mu = np.array([0.1] * N)  # Base intensities
K = 100  # Total grid points (50 linear + 50 log)

def phi(t):  # Initial kernel function
    return np.array([[0.1 * np.exp(-t)] * N] * N)

def psi(t):  # Initial response function
    return np.array([0.1 * np.exp(-t)] * N)

# Create output directories
os.makedirs("results/hawkes/plots", exist_ok=True) 
os.makedirs("results/hawkes/phi_values", exist_ok=True)

# Open log file
with open("results/hawkes/runinfo.txt", "a") as log_file:
    log_file.write(f"Run started at {datetime.now()}\n\n")

    # Process each stock
    stocks = ["GOOGL", "AAPL", "AMZN", "AAL", "MSFT", "GT", "INTC", "IOVA", "PTEN", "MLCO",
              "PTON", "VLY", "VOD", "CSX", "WB", "BGC", "GRAB", "KHC", "HLMN", "IEP",
              "GBDC", "WBD", "PSNY", "NTAP", "GEO", "LCID", "GCMG", "CXW", "RIOT", "HL",
              "CX", "ERIC", "UA"]
    
    stocks = ["UA"]

    for stock in tqdm(stocks, desc="Processing stocks"):
        data_path = f"{os.getenv('FOLDER_PATH')}/data/hawkes_dataset/{stock}"
        if not os.path.exists(data_path):
            log_file.write(f"Warning: Path {data_path} does not exist\n")
            continue
            
        for file in tqdm(os.listdir(data_path), desc=f"Processing {stock} files"):
            try:
                # Read data
                start_read = time.time()
                df = pl.read_parquet(f"{data_path}/{file}")
                
                read_time = time.time() - start_read
                log_file.write(f"\n{stock}/{file}:\n")
                log_file.write(f"Data reading time: {read_time:.2f}s\n")
                
                # Get min and max time deltas
                df = df.with_columns(pl.col("ts_event").diff().alias("delta_t"))
                delta_t_ns = df["delta_t"].cast(pl.Int64)  # Get nanoseconds
                T_min = float(delta_t_ns.min()) / 1e9  # Convert to seconds
                T_max = float(delta_t_ns.max()) / 1e9 * 100
                
                # Create custom time grid with linear and log spacing
                t_linear = np.linspace(T_min, T_max/100, 50)  # First half linear
                t_log = np.logspace(np.log10(T_max/100), np.log10(T_max), 50)  # Second half log
                t_grid = np.concatenate([t_linear, t_log])
                
                # Initialize Hawkes model with custom grid
                hawkes = Hawkes(phi=phi, psi=psi, mu=mu, N=N, stepsize=K)
                hawkes.t_grid = t_grid  # Override default grid
                
                # Estimate g and solve Wiener-Hopf
                print("Estimating g...")
                start_g = time.time()
                g_results = hawkes.get_g_from_parquet(df)
                g_time = time.time() - start_g
                log_file.write(f"G estimation time: {g_time:.2f}s\n")
                
                print("Solving phi...")
                start_phi = time.time()
                phi_values = hawkes.solve_phi_from_wiener_hopf(g_results)
                phi_time = time.time() - start_phi
                log_file.write(f"Phi estimation time: {phi_time:.2f}s\n")
                
                # Save phi values
                print("Saving phi values...")
                date = file.split(".")[0]
                np.save(f"results/hawkes/phi_values/{stock}_{date}_phi.npy", phi_values)
                
                # Plot phi matrix
                print("Plotting phi matrix...")
                
                # Create figure and axes grid
                fig, axes = plt.subplots(N, N, figsize=(20, 20))
                
                event_types = ["P(a)", "P(b)", "T(a)", "T(b)", "L(a)", "L(b)", "C(a)", "C(b)"]
                
                # Get time points from quadrature scheme
                t_points = hawkes.quadrature_points
                
                for i in range(N):
                    for j in range(N):
                        start_plot = time.time()
                        fig.suptitle(f"{stock} - {date} - Kernel Functions")
                        ax = axes[i,j]
                        
                        # Plot phi values at quadrature points
                        phi_at_points = np.array([hawkes.phi(t)[i,j] for t in t_points])
                        ax.plot(t_points, phi_at_points, 'o-')
                        
                        if i == N-1:
                            ax.set_xlabel("Time")
                        if j == 0:
                            ax.set_ylabel("Amplitude")
                        ax.set_title(f"{event_types[i]} → {event_types[j]}")
                        ax.grid(True)
                
                plt.tight_layout()
                plt.savefig(f"results/hawkes/plots/{stock}_{date}_kernels.png")
                plt.close()
                plot_time = time.time() - start_plot
                log_file.write(f"Plotting time: {plot_time:.2f}s\n")
                
            except Exception as e:
                print(f"Error processing {stock}/{file}: {str(e)}")
                log_file.write(f"Error processing {stock}/{file}: {str(e)}\n")
                continue

    log_file.write(f"\nRun completed at {datetime.now()}")

print("Processing complete!")
