# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# +
import os
import polars as pl
import dotenv
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import eigvalsh # Use eigvalsh for symmetric matrices like correlation matrices

# Load environment variables (ensure FOLDER_PATH is set in your .env file)
dotenv.load_dotenv()
FOLDER_PATH = os.getenv("FOLDER_PATH")

if FOLDER_PATH is None:
    raise ValueError("FOLDER_PATH environment variable not set. Please create a .env file with FOLDER_PATH.")
# -

# +
# --- Stock and Date Identification ---

stocks_list = [d for d in os.listdir(FOLDER_PATH) if os.path.isdir(os.path.join(FOLDER_PATH, d))]
print(f"Found {len(stocks_list)} potential stock folders.")

stock_files = {}
date_counts = {}
date_stocks = {}  # Dictionary to store stocks for each date

print("Scanning stock directories for parquet files...")
for stock in tqdm(stocks_list):
    stock_path = os.path.join(FOLDER_PATH, stock)
    try:
        files = [f for f in os.listdir(stock_path) if f.endswith('.parquet') and f.startswith(f"{stock}_")]
        if not files:
            print(f"Warning: No parquet files found for stock {stock}")
            continue
            
        files.sort()
        stock_files[stock] = set(files)

        # Count occurrences of each date and store stocks
        for file in files:
            try:
                # Extract date from filename (remove stock prefix and .parquet suffix)
                date_str = file.replace(f"{stock}_", "").replace(".parquet", "")
                # Basic validation of date format YYYY-MM-DD
                if len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
                    if date_str in date_counts:
                        date_counts[date_str] += 1
                        date_stocks[date_str].append(stock)
                    else:
                        date_counts[date_str] = 1
                        date_stocks[date_str] = [stock]
                else:
                     print(f"Warning: Skipping file with unexpected name format: {file}")
            except Exception as e:
                print(f"Warning: Error processing file name {file}: {e}")

    except FileNotFoundError:
        print(f"Warning: Folder not found for stock {stock}, skipping.")
    except Exception as e:
        print(f"Warning: Error accessing folder for stock {stock}: {e}")

if not date_counts:
    raise ValueError("No valid parquet files found or processed. Check FOLDER_PATH and file naming convention (STOCK_YYYY-MM-DD.parquet).")

# Find the most common date(s)
max_count = max(date_counts.values())
most_common_dates = [date for date, count in date_counts.items() if count == max_count]

print(f"\nMaximum number of stocks available on a single day: {max_count}")
print(f"Date(s) with maximum count: {most_common_dates}")
if most_common_dates:
    print(f"Stocks available on {most_common_dates[0]}: {len(date_stocks[most_common_dates[0]])}")
    # print(f"Stocks: {date_stocks[most_common_dates[0]]}") # Uncomment to see list of stocks
else:
    raise ValueError("Could not determine any common dates with stocks.")

target_dates = most_common_dates # Process all dates with the maximum count
# target_dates = most_common_dates[:1] # Or uncomment to process only the first one
# -

# +
# --- Helper Function: Curate Mid Price ---

def curate_mid_price(df: pl.DataFrame, stock: str) -> pl.DataFrame:
    """
    Cleans the raw stock data, filters time ranges, and calculates microprice.
    Handles potential missing columns gracefully.
    """
    required_cols = {"ts_event", "ask_px_00", "bid_px_00", "ask_sz_00", "bid_sz_00"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"Warning: Stock {stock} missing required columns: {missing}. Returning original DataFrame.")
        return df # Or handle differently, e.g., return empty DF or raise error

    # Filter by publisher_id if it exists
    if "publisher_id" in df.columns:
        try:
            publisher_counts = df.group_by("publisher_id").len().sort("len", descending=True)
            if len(publisher_counts) > 1 and publisher_counts["publisher_id"][0] == 41:
                df = df.filter(pl.col("publisher_id") == 41)
            elif len(publisher_counts) > 1:
                 print(f"Info: Stock {stock}: Most frequent publisher is not 41, using all publishers.")
        except Exception as e:
            print(f"Warning: Error filtering by publisher_id for stock {stock}: {e}")


    # Filter time range (adjust GOOGL time if needed based on market)
    try:
        if stock == "GOOGL": # Example: Different trading hours for GOOGL?
             df = df.filter((pl.col("ts_event").dt.hour() >= 13) & (pl.col("ts_event").dt.hour() < 20)) # Between 1 PM and 8 PM UTC? Adjust as needed
        else:
            # Standard US market hours 9:30 AM to 4 PM ET - adjust for UTC if data is in UTC
            # Example: Assuming data is UTC and market is ET (UTC-4 during EDT, UTC-5 during EST)
            # Let's assume UTC for simplicity, standard hours 9:30 to 16:00
             df = df.filter(
                 (pl.col("ts_event").dt.hour() * 60 + pl.col("ts_event").dt.minute() >= 9 * 60 + 30) & # >= 9:30
                 (pl.col("ts_event").dt.hour() < 16)                                                  # < 16:00
            )
            # Remove potential opening auction noise (first few minutes)
            df = df.filter(pl.col("ts_event").dt.time() > pl.time(9, 35, 0)) # Start after 9:35 AM

    except Exception as e:
         print(f"Warning: Error filtering time for stock {stock}: {e}. Proceeding with unfiltered time.")


    # Calculate microprice
    df = df.with_columns(
        micro_price = (pl.col("ask_px_00") * pl.col("bid_sz_00") + pl.col("bid_px_00") * pl.col("ask_sz_00"))
                       / (pl.col("ask_sz_00") + pl.col("bid_sz_00"))
    )

    # Handle potential division by zero or NaNs, fill with forward fill
    # Check for inf values resulting from calculation before filling NaNs
    df = df.with_columns(
        pl.when(pl.col("micro_price").is_infinite())
        .then(None) # Replace inf with null
        .otherwise(pl.col("micro_price"))
        .alias("micro_price")
    )
    df = df.fill_null(strategy="forward") # Forward fill NaNs first
    df = df.fill_nan(None) # Replace any remaining NaNs (e.g., at the start) with Null

    # Drop rows where microprice is still null after forward fill (usually only the first few rows)
    df = df.drop_nulls(subset=["micro_price"])

    # Ensure data is sorted by time
    df = df.sort("ts_event")

    # Keep only necessary columns
    df = df.select(["ts_event", "micro_price"])

    return df
# -

# +
# --- Main Processing Loop ---

target_time_scales = ["1m", "10ms"]

for date in target_dates:
    print(f"\n{'='*20} Processing Date: {date} {'='*20}")
    stocks_on_this_date = date_stocks[date]
    
    daily_data = {} # Store curated dataframes for the day
    print(f"Loading and curating data for {len(stocks_on_this_date)} stocks on {date}...")
    for stock in tqdm(stocks_on_this_date, desc=f"Curating {date}"):
        file_path = os.path.join(FOLDER_PATH, stock, f"{stock}_{date}.parquet")
        try:
            df_raw = pl.read_parquet(file_path)
            if df_raw.height == 0:
                print(f"Warning: Empty dataframe for {stock} on {date}. Skipping stock.")
                continue
            df_curated = curate_mid_price(df_raw, stock)
            if df_curated.height > 1: # Need at least 2 points to calculate returns
                daily_data[stock] = df_curated
            else:
                 print(f"Warning: Not enough data points after curating for {stock} on {date}. Skipping stock.")
        except Exception as e:
            print(f"Error loading or curating {stock} on {date}: {e}")
            
    if not daily_data:
        print(f"No valid data loaded for any stock on {date}. Skipping date.")
        continue
        
    print(f"Successfully curated data for {len(daily_data)} stocks.")

    for time_scale in target_time_scales:
        print(f"\n--- Processing Time Scale: {time_scale} ---")
        
        resampled_returns = {}
        print(f"Resampling to {time_scale} and calculating log returns...")
        
        base_df = None # To store the timestamp index for joining

        for stock, df in tqdm(daily_data.items(), desc=f"Resampling {time_scale}"):
            try:
                # Resample data
                df_resampled = df.group_by_dynamic(
                    index_column="ts_event",
                    every=time_scale,
                    closed='left' # include left endpoint, exclude right
                ).agg(
                    pl.col("micro_price").last().alias(stock) # Rename column to stock name
                )

                # Calculate log returns
                df_resampled = df_resampled.with_columns(
                    pl.col(stock).log().diff().alias(stock) # Calculate log return, keep stock name
                )

                # Drop the first row (which has null return) and any other nulls
                df_resampled = df_resampled.drop_nulls(subset=[stock])

                if df_resampled.height == 0:
                     print(f"Warning: No valid returns after resampling for {stock} at {time_scale}. Skipping stock.")
                     continue

                resampled_returns[stock] = df_resampled

                # Use the first stock's timestamps as the base for joining
                if base_df is None:
                     base_df = df_resampled.select("ts_event")

            except Exception as e:
                print(f"Error resampling or calculating returns for {stock} at {time_scale}: {e}")
                
        if not resampled_returns or base_df is None:
            print(f"Could not generate returns for any stock at timescale {time_scale}. Skipping.")
            continue

        # --- Align Data using outer join ---
        print("Aligning data across stocks...")
        aligned_df_pl = base_df # Start with the base timestamps

        for stock, df_ret in resampled_returns.items():
             # Ensure the stock column exists before joining
             if stock not in df_ret.columns:
                  print(f"Warning: Column '{stock}' not found in resampled returns for {stock}. Skipping join.")
                  continue
             # Select only ts_event and the stock return column
             df_to_join = df_ret.select(["ts_event", stock])
             try:
                  # Perform outer join to keep all timestamps and fill missing values with null
                  aligned_df_pl = aligned_df_pl.join(df_to_join, on="ts_event", how="outer", coalesce=True)
             except Exception as e:
                  print(f"Error joining data for stock {stock}: {e}")


        # Convert to Pandas for easier correlation and NaN handling per row
        try:
             aligned_df_pd = aligned_df_pl.to_pandas().set_index("ts_event")
        except Exception as e:
             print(f"Error converting aligned data to Pandas: {e}. Skipping timescale {time_scale}.")
             continue

        # Drop rows where ALL stock returns are NaN (less likely with outer join but good practice)
        aligned_df_pd = aligned_df_pd.dropna(how='all')
        # Drop columns that are entirely NaN
        aligned_df_pd = aligned_df_pd.dropna(axis=1, how='all')


        # Optional: Fill remaining NaNs or drop rows with any NaNs
        # Option 1: Forward fill (can introduce biases)
        # aligned_df_pd = aligned_df_pd.ffill()
        # Option 2: Drop rows with *any* NaN (might lose a lot of data)
        aligned_df_pd = aligned_df_pd.dropna(how='any')
        # Option 3: Interpolate (linear, time, etc.) - more complex
        # aligned_df_pd = aligned_df_pd.interpolate(method='time')


        if aligned_df_pd.shape[0] < 2 or aligned_df_pd.shape[1] < 2:
             print(f"Not enough data ({aligned_df_pd.shape[0]} rows, {aligned_df_pd.shape[1]} stocks) "
                   f"remaining after alignment and NaN handling for {time_scale}. Skipping correlation.")
             continue
             
        print(f"Aligned data shape for correlation ({time_scale}): {aligned_df_pd.shape} (rows, stocks)")
        final_stocks = aligned_df_pd.columns.tolist() # Stocks actually used

        # --- Calculate Correlation Matrix ---
        print("Calculating correlation matrix...")
        try:
            corr_matrix = aligned_df_pd.corr(method='pearson')
        except Exception as e:
             print(f"Error calculating correlation matrix: {e}. Skipping timescale {time_scale}.")
             continue

        # --- Plot Correlation Matrix Heatmap ---
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f",
                    xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns)
        plt.title(f"Correlation Matrix ({len(final_stocks)} Stocks) - {date} - {time_scale}")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        # plt.savefig(f"correlation_matrix_{date}_{time_scale}.png") # Optional: save figure

        # --- Random Matrix Theory (RMT) Analysis ---
        print("\nPerforming RMT analysis...")
        try:
            # Ensure matrix is symmetric and has no NaNs before eigenvalue calculation
            if not corr_matrix.isna().any().any() and np.allclose(corr_matrix.values, corr_matrix.values.T):
                eigenvalues = eigvalsh(corr_matrix) # Use eigvalsh for real symmetric matrices
                eigenvalues.sort() # Sort ascending
                
                T, N = aligned_df_pd.shape # Number of time points, Number of stocks
                if N <= 1:
                     print("Skipping RMT analysis: Need at least 2 stocks.")
                     continue
                Q = T / N
                
                if Q < 1:
                    print(f"Warning: RMT Marchenko-Pastur law requires T >= N (Q >= 1). Current Q={Q:.2f}. Bounds may not be reliable.")
                    # Still calculate bounds for reference, but with a warning
                
                # Marchenko-Pastur theoretical bounds (assuming variance sigma^2=1 for correlation matrix)
                lambda_plus = (1 + np.sqrt(1/Q))**2
                lambda_minus = (1 - np.sqrt(1/Q))**2 if Q >=1 else 0 # lambda_minus is 0 if Q < 1

                print(f"  Number of time points (T): {T}")
                print(f"  Number of stocks (N): {N}")
                print(f"  Q = T/N: {Q:.4f}")
                print(f"  Marchenko-Pastur Bounds: [{lambda_minus:.4f}, {lambda_plus:.4f}]")

                # Identify eigenvalues outside the theoretical bulk
                significant_eigenvalues = eigenvalues[eigenvalues > lambda_plus]
                print(f"  Number of eigenvalues: {len(eigenvalues)}")
                print(f"  Eigenvalues range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")
                print(f"  Number of eigenvalues > lambda_plus: {len(significant_eigenvalues)}")
                if len(significant_eigenvalues) > 0:
                    print(f"  Largest eigenvalues (potentially significant): {significant_eigenvalues[::-1][:5]}") # Show top 5 largest

                # --- Plot Eigenvalue Distribution ---
                plt.figure(figsize=(10, 6))
                plt.hist(eigenvalues, bins='auto', density=True, label='Empirical Eigenvalue Distribution')
                plt.axvline(lambda_minus, color='r', linestyle='--', label=f'λ- = {lambda_minus:.3f}')
                plt.axvline(lambda_plus, color='r', linestyle='--', label=f'λ+ = {lambda_plus:.3f}')
                plt.title(f'Eigenvalue Distribution vs Marchenko-Pastur - {date} - {time_scale}')
                plt.xlabel('Eigenvalue')
                plt.ylabel('Density')
                plt.legend()
                plt.tight_layout()
                plt.show()
                # plt.savefig(f"eigenvalue_distribution_{date}_{time_scale}.png") # Optional: save figure

            else:
                print("Could not compute eigenvalues: Correlation matrix contains NaNs or is not symmetric.")

        except Exception as e:
            print(f"Error during RMT analysis: {e}")

print("\nCorrelation analysis complete.")
# -
