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

# + endofcell="--"


import os
import polars as pl
import dotenv
from tqdm import tqdm
import plotly.graph_objects as go
FOLDER_PATH = os.getenv("FOLDER_PATH")
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.optimize import curve_fit


dotenv.load_dotenv()
# # +
# Get list of all stocks from .env
stocks_list = ["GOOGL", "AAPL", "AMZN", "AAL", "MSFT", "GT", "INTC", "IOVA", "PTEN", 
               "MLCO", "PTON", "VLY", "VOD", "CSX", "WB", "BGC", "GRAB", "KHC", "HLMN",
               "IEP", "GBDC", "WBD", "PSNY", "NTAP", "GEO", "LCID", "GCMG", "CXW", 
               "RIOT", "HL", "CX", "ERIC", "UA"]

# Get parquet files for each stock and count occurrences of each date
stock_files = {}
date_counts = {}
date_stocks = {}  # Dictionary to store stocks for each date
for stock in stocks_list:
    files = [f for f in os.listdir(f"{FOLDER_PATH}{stock}") if f.endswith('.parquet')]
    files.sort()
    stock_files[stock] = set(files)
    
    # Count occurrences of each date and store stocks
    for file in files:
        # Extract date from filename (remove stock prefix and .parquet suffix)
        date = file.replace(f"{stock}_", "").replace(".parquet", "")
        if date in date_counts:
            date_counts[date] += 1
            date_stocks[date].append(stock)
        else:
            date_counts[date] = 1
            date_stocks[date] = [stock]

# Find the most common date
print(date_counts)
print("Stocks for each date:", date_stocks)
most_common_date = max(date_counts.items(), key=lambda x: x[1])[0]
print(f"Most common date across stocks: {most_common_date}")

# -

#

# # +
# Get all dates with maximum count
max_count = max(date_counts.values())
most_common_dates = [date for date, count in date_counts.items() if count == max_count]
print(f"Dates with maximum count ({max_count} stocks): {most_common_dates}")
print("stocks for most common date:", date_stocks[most_common_dates[0]])
# Create empty list to store all dataframes
all_dfs = {stock: pl.DataFrame() for stock in date_stocks[most_common_dates[0]]}

# Load and combine data for each date
for date in most_common_dates[:1]:
    print(f"\nProcessing date: {date}")
    stocks_for_date = date_stocks[date]
    
    # Load data for each stock on this date
    for stock in tqdm(stocks_for_date):
        file_path = f"{FOLDER_PATH}{stock}/{stock}_{date}.parquet"
        if os.path.exists(file_path):
            df = pl.read_parquet(file_path)
            all_dfs[stock] = pl.concat([all_dfs[stock], df])





def curate_mid_price(df,stock):
    if "publisher_id" in df.columns:
        num_entries_by_publisher = df.group_by("publisher_id").len().sort("len", descending=True)
        if len(num_entries_by_publisher) > 1:
                df = df.filter(pl.col("publisher_id") == 41)
        
        
    if stock == "GOOGL":
        df = df.filter(pl.col("ts_event").dt.hour() >= 13)
        df = df.filter(pl.col("ts_event").dt.hour() <= 20)
        
        
    else:
        df = df.filter(
            (
                (pl.col("ts_event").dt.hour() == 9) & (pl.col("ts_event").dt.minute() >= 35) |
                (pl.col("ts_event").dt.hour() > 9) & (pl.col("ts_event").dt.hour() < 16)
            )
        )
    
    # Remove the first row at 9:30
    df = df.with_row_index("index").filter(
        ~((pl.col("ts_event").dt.hour() == 9) & 
          (pl.col("ts_event").dt.minute() == 30) & 
          (pl.col("index") == df.filter(
              (pl.col("ts_event").dt.hour() == 9) & 
              (pl.col("ts_event").dt.minute() == 30)
          ).with_row_index("index").select("index").min())
        )
    ).drop("index")
    mid_price = (df["ask_px_00"] + df["bid_px_00"]) / 2
    
    # managing nans or infs, preceding value filling
    mid_price = mid_price.fill_nan(mid_price.shift(1))
    df = df.with_columns(mid_price=mid_price)
    # sort by ts_event
    # added microprice
    microprice = (df["ask_px_00"]*df["bid_sz_00"] + df["bid_px_00"]*df["ask_sz_00"]) / (df["ask_sz_00"] + df["bid_sz_00"])
    # remove nans or infs
    microprice = microprice.fill_nan(microprice.shift(1))
    df = df.with_columns(microprice=microprice)
    df = df.sort("ts_event")
    return df


for stock in tqdm(date_stocks[most_common_dates[0]], "Huge amount of data to process"):
        df = all_dfs[stock]
        df  = curate_mid_price(df,stock)
        all_dfs[stock] = df


# --- Main Processing Loop ---

print("\nCalculating correlation matrices for microprice variations...")

# Define time scales for resampling
time_scales = ["10s", "1s", "10ms"]
time_scales.reverse()

for time_scale in time_scales:
    print(f"\nProcessing time scale: {time_scale}")
    
    # Sample each stock at regular intervals and calculate variations
    stock_variations = {}
    for stock in date_stocks[most_common_dates[0]]:
        print(f"Processing stock: {stock}")
        df = all_dfs[stock]
        
        # Resample data at regular intervals using group_by_dynamic
        sampled_prices = df.group_by_dynamic(
            "ts_event",
            every=time_scale
        ).agg([
            pl.col("microprice").last().alias("microprice")
        ])
        
        # Calculate microprice variations (differences)
        variations = np.diff(sampled_prices["microprice"].to_numpy())
        stock_variations[stock] = variations
        
        # Print first 5 values of the time series
        print(f"First 5 variations for {stock}:")
        print(variations[:5])
        print(f"Number of data points for {stock}: {len(variations)}")
    
    # Ensure all stocks have same number of variations
    min_length = min(len(variations) for variations in stock_variations.values())
    for stock in stock_variations:
        stock_variations[stock] = stock_variations[stock][:min_length]
    
    print(f"\nAll stocks trimmed to {min_length} data points")
    
    # Create correlation matrix
    stocks = list(stock_variations.keys())
    corr_matrix = np.zeros((len(stocks), len(stocks)))
    
    for i in range(len(stocks)):
        for j in range(len(stocks)):
            corr = np.corrcoef(stock_variations[stocks[i]], stock_variations[stocks[j]])[0,1]
            corr_matrix[i,j] = corr
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(stocks)), stocks, rotation=90)
    plt.yticks(range(len(stocks)), stocks)
    plt.title(f"Microprice Variations Correlation Matrix - {most_common_dates[0]} - {time_scale}")
    
    # Add correlation values in the center of each cell
    for i in range(len(stocks)):
        for j in range(len(stocks)):
            plt.text(j, i, f'{corr_matrix[i,j]:.2f}', 
                    ha='center', va='center', color='black')
    
    plt.tight_layout()
    plt.savefig(f"/home/janis/HFTP2/HFT/results/correlation/variations_matrix_{most_common_dates[0]}_{time_scale}.png")
    plt.show()

print("\nCorrelation analysis complete.")