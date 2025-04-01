

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




def run1(): 
    for stock in tqdm(date_stocks[most_common_dates[0]], "Huge amount of data to process"):
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

def run2():
    # for each day, we fit a copula for mid price variations depending on their stock
    for date in most_common_dates:
        date_df = pl.DataFrame()
        for stock in date_stocks[date]:
            df = all_dfs[stock]
            date_df = pl.concat([date_df, df])
        
        micro_variations = []
        # Sample at different time scales from 1us to 1000s
        time_scales = ["30us", "100us", "1ms", "10ms", "100ms", "1s"]
        time_scales.reverse()
        for time_scale in tqdm(time_scales, "Fitting copulas"):
            # we write in a txt the time scale
            with open(f"/home/janis/HFTP2/HFT/results/copulas/plots/time_scale_{time_scale}.txt", "a") as f:
                f.write(f"Time scale: {time_scale}")
            for stock in tqdm(date_stocks[date], "Fitting copulas"):
                df = all_dfs[stock]
                # Calculate returns instead of raw prices for better copula fitting
                micro_data = df.group_by_dynamic(
                    "ts_event",
                    every=time_scale
                ).agg([
                    pl.col("microprice").last().alias("microprice")
                ])
                returns = np.diff(np.log(micro_data["microprice"].to_numpy()))
                micro_variations.append(returns)
            fraction = 0.01
            micro_variations = micro_variations[:int(len(micro_variations)*fraction)]
            
            # Fit different types of copulas
            from copulalib.copulalib import Copula
            from statsmodels.distributions.copula.api import GaussianCopula, StudentTCopula
            
            copula_types = {
                'Gaussian': ('gaussian', GaussianCopula()),
                'Student-t': ('student', StudentTCopula()),
                'Clayton': ('clayton', None),
                'Frank': ('frank', None),
                'Gumbel': ('gumbel', None)
            }
            
            print("Now fitting copulas")
            # Calculate and plot each copula
            for copula_name, (family, statsmodel_copula) in tqdm(copula_types.items(), "Fitting copulas"):
                try:
                    # Check if we have enough data points
                    if len(micro_variations) < 2:
                        print(f"Warning: Not enough data points for {copula_name} copula")
                        continue
                        
                    if len(micro_variations[0]) == 0 or len(micro_variations[1]) == 0:
                        print(f"Warning: Empty data arrays for {copula_name} copula")
                        continue
                        
                    # Open files in append mode at the start
                    stats_file = open(f"/home/janis/HFTP2/HFT/results/copulas/stats/{copula_name}_{date}_{time_scale}.txt", "a")
                    
                    if statsmodel_copula is None:
                        # Use copulalib implementation
                        try:
                            copula = Copula(micro_variations[0], micro_variations[1], family=family)
                        except IndexError:
                            print(f"Warning: Index error when creating {copula_name} copula")
                            stats_file.close()
                            continue
                            
                        # Calculate correlations
                        kendall_tau = copula.tau
                        spearman_corr = copula.sr
                        pearson_corr = copula.pr
                        theta = copula.theta
                        
                        # Generate samples for plotting
                        X1, Y1 = copula.generate_xy(1000)
                    else:
                        # Create new instance for each fit to avoid attribute issues
                        copula = type(statsmodel_copula)()
                        
                        # Check for sufficient variation in data
                        if len(set(micro_variations[0])) < 2 or len(set(micro_variations[1])) < 2:
                            print(f"Warning: Insufficient variation in data for {copula_name} copula")
                            stats_file.write(f"Warning: Insufficient variation in data for {copula_name} copula\n")
                            stats_file.close()
                            continue
                            
                        try:
                            copula.fit(micro_variations)
                        except (IndexError, ValueError) as e:
                            print(f"Warning: Error fitting {copula_name} copula: {e}")
                            stats_file.close()
                            continue
                            
                        # Calculate correlations
                        kendall_tau = copula.kendall_tau()
                        spearman_corr = copula.spearman_correlation()
                        pearson_corr = copula.pearson_correlation()
                        theta = copula.params[0] if hasattr(copula, 'params') else None
                        
                        # Generate samples for plotting
                        samples = copula.random(1000)
                        X1, Y1 = samples[:, 0], samples[:, 1]
                    
                    # Print and write results in real-time
                    stats_file.write(f"\nResults for {copula_name} copula at {time_scale} scale:\n")
                    if theta is not None:
                        print(f"Theta: {theta:.3f}")
                        stats_file.write(f"Theta: {theta:.3f}\n")
                    print(f"Kendall tau: {kendall_tau:.3f}")
                    stats_file.write(f"Kendall tau: {kendall_tau:.3f}\n")
                    print(f"Spearman correlation: {spearman_corr:.3f}")
                    stats_file.write(f"Spearman correlation: {spearman_corr:.3f}\n")
                    print(f"Pearson correlation: {pearson_corr:.3f}")
                    stats_file.write(f"Pearson correlation: {pearson_corr:.3f}\n")
                    stats_file.flush()  # Force write to disk
                    
                    # Plot
                    plt.figure(figsize=(12, 8))
                    plt.scatter(X1, Y1, alpha=0.5)
                    plt.xlabel('U')
                    plt.ylabel('V')
                    
                    # Add correlation info to plot
                    info_text = [
                        f'Kendall tau: {kendall_tau:.3f}',
                        f'Spearman corr: {spearman_corr:.3f}',
                        f'Pearson corr: {pearson_corr:.3f}'
                    ]
                    if theta is not None:
                        info_text.insert(0, f'Theta: {theta:.3f}')
                    
                    plt.text(0.05, 0.95,
                            '\n'.join(info_text),
                            transform=plt.gca().transAxes,
                            bbox=dict(facecolor='white', alpha=0.8))
                    
                    plt.title(f"{copula_name} Copula - {date} - Scale {time_scale}")
                    plt.savefig(f"/home/janis/HFTP2/HFT/results/copulas/plots/{copula_name}_copula_{date}_{time_scale}.png")
                    plt.close()
                    
                    stats_file.close()
                
                except Exception as e:
                    print(f"Error fitting {copula_name} copula: {e}")
                    with open(f"/home/janis/HFTP2/HFT/results/copulas/stats/{copula_name}_{date}_{time_scale}.txt", "a") as f:
                        f.write(f"Error fitting copula: {e}\n")
                    continue

if __name__ == "__main__":
    #run1()
    run2()
