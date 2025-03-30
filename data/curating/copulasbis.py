

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
# +
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

# +
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
    df = all_dfs[stock]
    # average bid ask spread
    avg_spread = (df["ask_px_00"] - df["bid_px_00"]).mean()
    print(f"Average bid ask spread: {avg_spread}")
    # Calculate time differences between mid price changes in nanoseconds and convert to milliseconds
    time_diffs = df.with_columns(
        mid_price_change=pl.col("mid_price").diff()
    ).filter(
        pl.col("mid_price_change") != 0
    ).select(
        (pl.col("ts_event").diff().cast(pl.Int64) / 1_000_000).alias("time_diff_ms")  # Convert to milliseconds
    ).drop_nulls()

    # Filter out times > 1 hour (3600000 milliseconds) 
    time_diffs = time_diffs.filter(pl.col("time_diff_ms") <= 36000)
    alpha = 0.5  # Use first 10% of data
    time_diffs_np = time_diffs.to_numpy().flatten()[:int(len(time_diffs) * alpha)]
    avg_arrival_time = time_diffs.mean()["time_diff_ms"][0] 
    time_scales = [str(int(k*avg_arrival_time))+"us" for k in [1,5,10,30,100,1000,3000,10000,30000,100000,300000,1000000,3000000]]
    print(time_scales)

    time_scales = time_scales

    dfs = {}

    for scale in time_scales:
        
        df_temp = df.group_by(pl.col("ts_event").dt.truncate(scale)).agg([
            pl.col("mid_price").last().alias("mid_price")
        ])
        
        df_temp = df_temp.sort("ts_event")
        
        df_temp = df_temp.with_columns(
            tick_variation=pl.when(pl.col("ts_event").dt.date().diff() == 0)
            .then(pl.col("mid_price").diff()/avg_spread)
            .otherwise(None)
        )
        df_temp = df_temp.with_columns(
            log_variation=pl.when(pl.col("ts_event").dt.date().diff() == 0)
            .then(pl.col("mid_price").log().diff())
            .otherwise(None)
        )
        
        dfs[scale] = df_temp
        

    def rational_func(x, a, b, c):
        return a / (b + np.power(np.abs(x), c))

    def plot_hist_with_gaussian(data, title):
        data_np = data.to_numpy()
        data_clean = data_np[~np.isnan(data_np) & ~np.isinf(data_np)]
        mu, std = norm.fit(data_clean)
        
        plt.figure(figsize=(10, 6))
        counts, bins, _ = plt.hist(data_clean, bins='auto', density=True, alpha=0.7)
        
        x = np.linspace(min(data_clean), max(data_clean), 100)
        y = norm.pdf(x, mu, std)
        plt.plot(x, y, 'r-', lw=2, label=f'Gaussian fit (μ={mu:.3f}, σ={std:.3f})')
        
        # Fit rational function to the positive side of the distribution
        bin_centers = (bins[:-1] + bins[1:]) / 2
        mask = (bin_centers > 0) & (counts > 0)
        if np.any(mask):
            try:
                popt, _ = curve_fit(rational_func, bin_centers[mask], counts[mask], p0=[1, 1, 2])
            except Exception as e:
                print(f"Error fitting rational function: {e}")
                popt = [np.nan, np.nan, np.nan]
            x_rational = np.linspace(max(min(data_clean), 0.01), max(data_clean), 100)
            y_rational = rational_func(x_rational, *popt)
            plt.plot(x_rational, y_rational, 'k-', lw=2, label=f'Rational fit (a={popt[0]:.3f}, b={popt[1]:.3f}, c={popt[2]:.3f})')
        
        plt.title(title)
        plt.xlabel('Spread Variation')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"/home/janis/HFTP2/HFT/results/copulas/plots/{stock}_{scale}_returns_histogram.png")
    
    for scale in time_scales:
        df_current = dfs[scale]
        title = f"Histogram of spread Variations - {scale} Sampling"
        plot_hist_with_gaussian(df_current["tick_variation"], title)



