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
import plotly.graph_objects as go
FOLDER_PATH = os.getenv("FOLDER_PATH")


dotenv.load_dotenv()
stock = "WBD"

# +
parquet_files = [f for f in os.listdir(f"{FOLDER_PATH}{stock}") if f.endswith('.parquet')]
parquet_files.sort()
print(len(parquet_files),"\n",parquet_files)
threshold = len(parquet_files)//10

parquet_files = parquet_files[:threshold]
# Read and concatenate all parquet files
df = pl.concat([
    pl.read_parquet(f"{FOLDER_PATH}{stock}/{file}") 
    for file in parquet_files
])


# -

def curate_mid_price(df,stock):
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
    # now we define the mid price with the microprice, barycenter of bid and ask prices by their weights
    
    micro_price = (df["ask_px_00"] * df["bid_sz_00"] + df["bid_px_00"] * df["ask_sz_00"]) / (df["bid_sz_00"] + df["ask_sz_00"])
    df = df.with_columns(mid_price=micro_price)
    # sort by ts_event
    df = df.sort("ts_event")
    return df


# +
df  = curate_mid_price(df,stock)

# average bid ask spread
avg_spread = (df["ask_px_00"] - df["bid_px_00"]).mean()
# -

print(f"Average bid ask spread: {avg_spread}")

df.head()

df_cleaned = df[["ts_event","mid_price"]]

df_cleaned = df[["ts_event","mid_price","micro_price"]]



# Average arrival time

# +
# Compute average time between mid price changes
import numpy as np
import matplotlib.pyplot as plt

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

# Take first alpha fraction of data
alpha = 0.1  # Use first 10% of data
time_diffs_np = time_diffs.to_numpy().flatten()[:int(len(time_diffs) * alpha)]

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(time_diffs_np, bins='auto', density=True, alpha=0.7)

plt.title(f'Distribution of Time Between Mid Price Changes (<1h) for {stock} (First {alpha*100}% of data)')
plt.xlabel('Time between mid price changes (milliseconds)')
plt.ylabel('Density')
plt.ylim(0,0.0002)
print('Average time between mid price changes:', time_diffs.mean())
avg_arrival_time = time_diffs.mean()["time_diff_ms"][0] 
plt.grid(True, alpha=0.3)

# +

# Save plot
os.makedirs(f"/home/janis/HFTP2/HFT/results/hurst/plots/{stock}/", exist_ok=True)
plt.savefig(f"/home/janis/HFTP2/HFT/results/hurst/plots/{stock}/{stock}_arrival_times.png")

# -

time_scales = [str(int(k*avg_arrival_time))+"us" for k in [1,5,10,30,100,1000,3000,10000,30000,100000,300000,1000000,3000000]]
print(time_scales)

# +
time_scales = time_scales

dfs = {}

for scale in time_scales:
    df_temp = df_cleaned.group_by(pl.col("ts_event").dt.truncate(scale)).agg([
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
    
    print(f"\n{scale} sampling:")
    print(df_temp.head())
# -

"""
import plotly.graph_objects as go

# Create plots for each time scale
for scale in time_scales:
    df_current = dfs[scale]
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df_current["ts_event"], y=df_current["mid_price"], name="Mid Price")
    )
    fig.update_layout(
        title=f"{scale} Sampling",
        xaxis_title="Time", 
        yaxis_title="Mid Price"
    )
    fig.show()
"""

# Calculate volatility under Bachelier model
# In Bachelier model, log returns are normally distributed with variance = sigma^2 * dt
# So sigma = sqrt(var(log returns) / dt)
for scale in time_scales:
    df_current = dfs[scale]
    
    # Add time difference column in seconds by converting nanoseconds to seconds
    df_current = df_current.with_columns(
        dt=pl.col("ts_event").diff().cast(pl.Int64) / 1e9  # Convert directly to seconds
    )
    
    # Calculate volatility using log variations and time differences, normalized by avg_spread
    df_current = df_current.with_columns(
        vol_bachelier=pl.when(pl.col("tick_variation").is_not_null())
        .then((pl.col("tick_variation").abs() / pl.col("dt").sqrt()) / avg_spread)
        .otherwise(None)
    )
    
    dfs[scale] = df_current
    
    # Print summary statistics
    print(f"\nBachelier volatility stats for {scale} sampling:")
    print(df_current.select("vol_bachelier").describe())


# +
# Plot histograms and time series of volatility for each sampling scale
import matplotlib.pyplot as plt

# Create subplots for histograms
fig_hist, axes_hist = plt.subplots(len(time_scales), 1, figsize=(10, 4*len(time_scales)))
fig_hist.suptitle('Volatility Histograms by Sampling Scale')

# Create subplots for time series
fig_ts, axes_ts = plt.subplots(len(time_scales), 1, figsize=(10, 4*len(time_scales)))
fig_ts.suptitle('Volatility Time Series by Sampling Scale')

for i, scale in enumerate(time_scales):
    df_current = dfs[scale]
    
    # Plot histogram
    vol_data = df_current.select('vol_bachelier').to_numpy().flatten()
    axes_hist[i].hist(vol_data[~np.isnan(vol_data)], bins=50, density=True)  # Remove NaN values
    axes_hist[i].set_title(f'{scale} Sampling')
    axes_hist[i].set_xlabel('Volatility')
    axes_hist[i].set_ylabel('Density')
    
    # Plot time series - first day only
    first_day = df_current.select('ts_event').to_numpy()[0].astype('datetime64[D]')
    mask = df_current.select('ts_event').to_numpy().astype('datetime64[D]') == first_day
    
    axes_ts[i].plot(df_current.select('ts_event').to_numpy()[mask], 
                    df_current.select('vol_bachelier').to_numpy()[mask])
    axes_ts[i].set_title(f'{scale} Sampling - First Day')
    axes_ts[i].set_xlabel('Time') 
    axes_ts[i].set_ylabel('Volatility')

plt.tight_layout()
plt.show()
plt.savefig(f"/home/janis/HFTP2/HFT/results/hurst/plots/{stock}/{stock}_volatility_histograms_{threshold}.png")
# -

# Calculate volatility under Bachelier model
# In Bachelier model, log returns are normally distributed with variance = sigma^2 * dt
# So sigma = sqrt(var(log returns) / dt)
for scale in time_scales:
    df_current = dfs[scale]
    
    # Add time difference column in seconds by converting nanoseconds to seconds
    df_current = df_current.with_columns(
        dt=pl.col("ts_event").diff().cast(pl.Int64) / 1e9  # Convert directly to seconds
    )
    
    # Calculate volatility using log variations and time differences, normalized by avg_spread
    df_current = df_current.with_columns(
        vol_bachelier=pl.when(pl.col("log_variation").is_not_null())
        .then((pl.col("log_variation").abs() / pl.col("dt").sqrt()) / avg_spread)
        .otherwise(None)
    )
    
    dfs[scale] = df_current
    
    # Print summary statistics
    print(f"\nBachelier volatility stats for {scale} sampling:")
    print(df_current.select("log_variation").describe())


# +
# Plot histograms and time series of volatility for each sampling scale
import matplotlib.pyplot as plt

# Create subplots for histograms
fig_hist, axes_hist = plt.subplots(len(time_scales), 1, figsize=(10, 4*len(time_scales)))
fig_hist.suptitle('Volatility Histograms by Sampling Scale')

# Create subplots for time series
fig_ts, axes_ts = plt.subplots(len(time_scales), 1, figsize=(10, 4*len(time_scales)))
fig_ts.suptitle('Volatility Time Series by Sampling Scale')

for i, scale in enumerate(time_scales):
    df_current = dfs[scale]
    
    # Plot histogram
    vol_data = df_current.select('log_variation').to_numpy().flatten()
    axes_hist[i].hist(vol_data[~np.isnan(vol_data)], bins=50, density=True)  # Remove NaN values
    axes_hist[i].set_title(f'{scale} Sampling')
    axes_hist[i].set_xlabel('Volatility')
    axes_hist[i].set_ylabel('Density')
    
    # Plot time series - first day only
    first_day = df_current.select('ts_event').to_numpy()[0].astype('datetime64[D]')
    mask = df_current.select('ts_event').to_numpy().astype('datetime64[D]') == first_day
    
    axes_ts[i].plot(df_current.select('ts_event').to_numpy()[mask], 
                    df_current.select('log_variation').to_numpy()[mask])
    axes_ts[i].set_title(f'{scale} Sampling - First Day')
    axes_ts[i].set_xlabel('Time') 
    axes_ts[i].set_ylabel('Volatility')

plt.tight_layout()
plt.show()
plt.savefig(f"/home/janis/HFTP2/HFT/results/hurst/plots/{stock}/{stock}_volatility_histograms_{threshold}.png")
plt.close("all")
# -

for scale in time_scales:
    # describe all columns
    print(dfs[scale].describe())


