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



# --

# for each day, we fit a copula for mid price variations depending on their stock
for date in most_common_dates:
    print(f"\nProcessing date: {date}")
    
    # Sample at different time scales
    time_scales = ["30us", "100us", "1ms", "10ms", "100ms", "1s"]
    time_scales.reverse()
    
    for time_scale in time_scales:
        print(f"\nProcessing time scale: {time_scale}")
        
        # Sample each stock at regular intervals and calculate returns
        stock_returns = {}
        for stock in date_stocks[date]:
            print(f"Processing stock: {stock}")
            df = all_dfs[stock]
            
            # Resample data at regular intervals using group_by_dynamic
            sampled_prices = df.group_by_dynamic(
                "ts_event",
                every=time_scale
            ).agg([
                pl.col("microprice").last().alias("microprice")
            ])
            
            # Calculate log returns
            returns = np.diff(np.log(sampled_prices["microprice"].to_numpy()))
            stock_returns[stock] = returns
            print(f"Number of data points for {stock}: {len(returns)}")
        
        # Ensure all stocks have same number of returns
        min_length = min(len(returns) for returns in stock_returns.values())
        for stock in stock_returns:
            stock_returns[stock] = stock_returns[stock][:min_length]
        
        print(f"\nAll stocks trimmed to {min_length} data points")
        
        # Fit different types of copulas
        from copulalib.copulalib import Copula
        
        copula_types = {
            'Clayton': 'clayton',
            'Frank': 'frank', 
            'Gumbel': 'gumbel'
        }
        
        # Store coefficients for all pairs
        correlation_results = []
        
        print("\nFitting copulas...")
        stocks = list(stock_returns.keys())
        for i in range(len(stocks)):
            for j in range(i+1, len(stocks)):
                stock1, stock2 = stocks[i], stocks[j]
                print(f"\nAnalyzing pair: {stock1} - {stock2}")
                
                for copula_name, family in copula_types.items():
                    try:
                        copula = Copula(stock_returns[stock1], stock_returns[stock2], family=family)
                        
                        # Store results in a dictionary
                        result = {
                            'stock1': stock1,
                            'stock2': stock2,
                            'copula_type': copula_name,
                            'theta': copula.theta,
                            'kendall_tau': copula.tau,
                            'spearman_corr': copula.sr,
                            'pearson_corr': copula.pr,
                            'X1': copula.generate_xy(1000)[0],
                            'Y1': copula.generate_xy(1000)[1]
                        }
                        correlation_results.append(result)
                        
                    except Exception as e:
                        print(f"Error fitting {copula_name} copula: {e}")
                        continue
        
        # Sort results by absolute value of Kendall's tau
        correlation_results.sort(key=lambda x: abs(x['kendall_tau']), reverse=True)
        
        # Plot and save only top 5 most correlated pairs
        for result in correlation_results[:5]:
            plt.figure(figsize=(12, 8))
            plt.scatter(result['X1'], result['Y1'], alpha=0.5)
            plt.xlabel('U')
            plt.ylabel('V')
            
            info_text = [
                f'Theta: {result["theta"]:.3f}',
                f'Kendall tau: {result["kendall_tau"]:.3f}',
                f'Spearman corr: {result["spearman_corr"]:.3f}',
                f'Pearson corr: {result["pearson_corr"]:.3f}'
            ]
            
            plt.text(0.05, 0.95,
                    '\n'.join(info_text),
                    transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))
            
            plt.title(f"{result['copula_type']} Copula - {result['stock1']} vs {result['stock2']}\n{date} - Scale {time_scale}")
            plt.savefig(f"/home/janis/HFTP2/HFT/results/copulas/stats/TOP_{result['stock1']}_{result['stock2']}_{date}_{time_scale}.png")
            plt.close()
            
            print(f"\nTop correlation pair: {result['stock1']} - {result['stock2']}")
            print(f"Copula type: {result['copula_type']}")
            print(f"Kendall tau: {result['kendall_tau']:.3f}")
            print(f"Spearman correlation: {result['spearman_corr']:.3f}")
            print(f"Pearson correlation: {result['pearson_corr']:.3f}")
