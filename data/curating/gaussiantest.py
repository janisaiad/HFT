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
stock = "CXW"
# -

threshold = 3
parquet_files = [f for f in os.listdir(f"{FOLDER_PATH}{stock}") if f.endswith('.parquet')]
parquet_files.sort()
print(parquet_files)
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

# +

# Create figure
fig = go.Figure()

# Add best bid line
fig.add_trace(go.Scatter(
    x=df['ts_event'],
    y=df['bid_px_00'],
    mode='lines',
    name='Best Bid',
    line=dict(color='blue')
))

# Add best ask line  
fig.add_trace(go.Scatter(
    x=df['ts_event'], 
    y=df['ask_px_00'],
    mode='lines',
    name='Best Ask',
    line=dict(color='red')
))

fig.add_trace(go.Scatter(
    x=df['ts_event'],
    y=df["mid_price"],
    mode='lines',
    name='Mid Price',
    line=dict(color='black')
))




# Update layout
fig.update_layout(
    title='Order Book and bid/ask',
    xaxis_title='Time',
    yaxis_title='Price',
    showlegend=True
)

fig.show()
# -

df_cleaned = df[["ts_event","mid_price"]]

# +
# Create different time-based resampled dataframes
df_30s = df_cleaned.group_by(pl.col("ts_event").dt.truncate("30s")).agg([
    pl.col("mid_price").last().alias("mid_price")
])


df_1min = df_cleaned.group_by(pl.col("ts_event").dt.truncate("1m")).agg([
    pl.col("mid_price").last().alias("mid_price")
])

df_5min = df_cleaned.group_by(pl.col("ts_event").dt.truncate("5m")).agg([
    pl.col("mid_price").last().alias("mid_price")
])

df_10min = df_cleaned.group_by(pl.col("ts_event").dt.truncate("10m")).agg([
    pl.col("mid_price").last().alias("mid_price")
])

# sorting by ts_event

df_30s = df_30s.sort("ts_event")
df_1min = df_1min.sort("ts_event")
df_5min = df_5min.sort("ts_event")

# tick variation in spread

df_30s = df_30s.with_columns(tick_variation=pl.col("mid_price").diff()/avg_spread)
df_30s = df_30s.with_columns(log_variation=pl.col("mid_price").log().diff())

df_1min = df_1min.with_columns(tick_variation=pl.col("mid_price").diff()/avg_spread)
df_1min = df_1min.with_columns(log_variation=pl.col("mid_price").log().diff())

df_5min = df_5min.with_columns(tick_variation=pl.col("mid_price").diff()/avg_spread)
df_5min = df_5min.with_columns(log_variation=pl.col("mid_price").log().diff())

df_10min = df_10min.with_columns(tick_variation=pl.col("mid_price").diff()/avg_spread)
df_10min = df_10min.with_columns(log_variation=pl.col("mid_price").log().diff())


print("\n30 seconds sampling:")
print(df_30s.head())
print("\n1 minute sampling:")
print(df_1min.head())
print("\n5 minutes sampling:")
print(df_5min.head())
print("\n10 minutes sampling:")
print(df_10min.head())


# +
import plotly.graph_objects as go

# 30 Seconds sampling plot
fig1 = go.Figure()
fig1.add_trace(
    go.Scatter(x=df_30s["ts_event"], y=df_30s["mid_price"], name="Mid Price")
)
fig1.update_layout(
    title="30 Seconds Sampling",
    xaxis_title="Time",
    yaxis_title="Mid Price"
)
fig1.show()

# 1 Minute sampling plot
fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(x=df_1min["ts_event"], y=df_1min["mid_price"], name="Mid Price")
)
fig2.update_layout(
    title="1 Minute Sampling",
    xaxis_title="Time",
    yaxis_title="Mid Price"
)
fig2.show()

# 5 Minutes sampling plot
fig3 = go.Figure()
fig3.add_trace(
    go.Scatter(x=df_5min["ts_event"], y=df_5min["mid_price"], name="Mid Price")
)
fig3.update_layout(
    title="5 Minutes Sampling", 
    xaxis_title="Time",
    yaxis_title="Mid Price"
)
fig3.show()

# 10 Minutes sampling plot
fig4 = go.Figure()
fig4.add_trace(
    go.Scatter(x=df_10min["ts_event"], y=df_10min["mid_price"], name="Mid Price")
)
fig4.update_layout(
    title="10 Minutes Sampling",
    xaxis_title="Time",
    yaxis_title="Mid Price"
)
fig4.show()


# +
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def plot_hist_with_gaussian(data, title):
    data_np = data.to_numpy()
    data_clean = data_np[~np.isnan(data_np) & ~np.isinf(data_np)]
    mu, std = norm.fit(data_clean)
    
    plt.figure(figsize=(10, 6))
    counts, bins, _ = plt.hist(data_clean, bins='auto', density=True, alpha=0.7)
    
    x = np.linspace(min(data_clean), max(data_clean), 100)
    y = norm.pdf(x, mu, std)
    plt.plot(x, y, 'r-', lw=2, label=f'Gaussian fit (μ={mu:.3f}, σ={std:.3f})')
    
    plt.title(title)
    plt.xlabel('Spread Variation')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()



# -

plot_hist_with_gaussian(df_30s["tick_variation"], "Histogram of spread Variations - 20 Seconds Sampling")
plot_hist_with_gaussian(df_1min["tick_variation"], "Histogram of spread Variations - 1 Minute Sampling")
plot_hist_with_gaussian(df_5min["tick_variation"], "Histogram of spread Variations - 5 Minutes Sampling")


#




