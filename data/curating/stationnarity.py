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
from ruptures import Window
from statsmodels.tsa.stattools import adfuller


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


# --

# For each stock and time scale, perform stationarity tests and detect regime changes
time_scales = ["30us", "100us", "1ms", "10ms", "100ms", "1s"]
time_scales.reverse()

for date in most_common_dates:
    print(f"\nProcessing date: {date}")
    
    for stock in date_stocks[date]:
        print(f"\nAnalyzing stock: {stock}")
        df = all_dfs[stock]
        
        for time_scale in time_scales:
            print(f"\nProcessing time scale: {time_scale}")
            
            # Resample data
            sampled_prices = df.group_by_dynamic(
                "ts_event",
                every=time_scale
            ).agg([
                pl.col("microprice").last().alias("microprice")
            ])
            
            prices = sampled_prices["microprice"].to_numpy()
            timestamps = sampled_prices["ts_event"].to_numpy()
            
            # Perform ADF test for stationarity
            adf_result = adfuller(prices)
            is_stationary = adf_result[1] < 0.05
            
            # Detect change points using Window method
            model = Window(width=50, model="l2").fit(prices.reshape(-1, 1))
            change_points = model.predict(pen=np.std(prices))
            
            # Plot the results
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Price series with change points
            plt.subplot(2, 1, 1)
            plt.plot(timestamps, prices, label='Price', alpha=0.7)
            for change_point in change_points:
                plt.axvline(x=timestamps[change_point], color='r', linestyle='--', alpha=0.5)
            plt.title(f"{stock} - {time_scale} - Price Series with Regime Changes\n" + 
                     f"Stationary: {is_stationary} (p-value: {adf_result[1]:.4f})")
            plt.legend()
            
            # Plot 2: Returns distribution
            plt.subplot(2, 1, 2)
            returns = np.diff(np.log(prices))
            plt.hist(returns, bins=50, density=True, alpha=0.7)
            plt.title(f"Returns Distribution")
            
            plt.tight_layout()
            plt.savefig(f"/home/janis/HFTP2/HFT/results/stationarity/stats/{stock}_{date}_{time_scale}.png")
            plt.close()
            
            print(f"ADF test p-value: {adf_result[1]}")
            print(f"Number of regime changes detected: {len(change_points)}")
