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
# -

parquet_files = [f for f in os.listdir(f"{FOLDER_PATH}{stock}") if f.endswith('.parquet')]
parquet_files.sort()
print(len(parquet_files),"\n",parquet_files)
threshold = len(parquet_files)//5
parquet_files = parquet_files[:threshold]
# Read and concatenate all parquet files
df = pl.concat([
    pl.read_parquet(f"{FOLDER_PATH}{stock}/{file}") 
    for file in parquet_files
])


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
    # sort by ts_event
    df = df.sort("ts_event")
    return df


# +
df  = curate_mid_price(df,stock)

# average bid ask spread
avg_spread = (df["ask_px_00"] - df["bid_px_00"]).mean()
# -

print(f"Average bid ask spread: {avg_spread}")

df_cleaned = df[["ts_event","mid_price"]]

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

# +
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


# +
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

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
        popt, _ = curve_fit(rational_func, bin_centers[mask], counts[mask], p0=[1, 1, 2])
        x_rational = np.linspace(max(min(data_clean), 0.01), max(data_clean), 100)
        y_rational = rational_func(x_rational, *popt)
        plt.plot(x_rational, y_rational, 'k-', lw=2, label=f'Rational fit (a={popt[0]:.3f}, b={popt[1]:.3f}, c={popt[2]:.3f})')
    
    plt.title(title)
    plt.xlabel('Spread Variation')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.savefig(f"/home/janis/HFTP2/HFT/results/hurst/plots/{stock}_{scale}_returns_histogram.png")



# -

for scale in time_scales:
    df_current = dfs[scale]
    title = f"Histogram of spread Variations - {scale} Sampling"
    plot_hist_with_gaussian(df_current["tick_variation"], title)



