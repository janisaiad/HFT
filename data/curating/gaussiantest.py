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
stock = "KHC"

# +
threshold = 1
parquet_files = [f for f in os.listdir(f"{FOLDER_PATH}{stock}") if f.endswith('.parquet')][:threshold]

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
                (pl.col("ts_event").dt.hour() == 9) & (pl.col("ts_event").dt.minute() >= 30) |
                (pl.col("ts_event").dt.hour() > 9) & (pl.col("ts_event").dt.hour() < 16) |
                (pl.col("ts_event").dt.hour() == 16) & (pl.col("ts_event").dt.minute() == 0)
            )
        )
    
    mid_price = (df["ask_px_00"] + df["bid_px_00"]) / 2
    
    # managing nans or infs, preceding value filling
    mid_price = mid_price.fill_nan(mid_price.shift(1))
    df = df.with_columns(mid_price=mid_price)
    # sort by ts_event
    df = df.sort("ts_event")
    return df


df  = curate_mid_price(df,stock)

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


df_30s = df_30s.with_columns(tick_variation=pl.col("mid_price").diff())
df_30s = df_30s.with_columns(log_variation=pl.col("mid_price").log().diff())

df_1min = df_1min.with_columns(tick_variation=pl.col("mid_price").diff())
df_1min = df_1min.with_columns(log_variation=pl.col("mid_price").log().diff())

df_5min = df_5min.with_columns(tick_variation=pl.col("mid_price").diff())
df_5min = df_5min.with_columns(log_variation=pl.col("mid_price").log().diff())



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
# Plot histograms with Gaussian fit for each sampling frequency
import numpy as np
from scipy.stats import norm

# Helper function to plot histogram with Gaussian fit
def plot_hist_with_gaussian(data, title):
    # Convert polars series to numpy array
    data_np = data.to_numpy()
    
    # Remove any infinite or NaN values
    data_clean = data_np[~np.isnan(data_np) & ~np.isinf(data_np)]
    
    # Fit normal distribution
    mu, std = norm.fit(data_clean)
    
    # Create histogram
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=data_clean,
        name="Log variations", 
        nbinsx=50,
        histnorm='probability density'
    ))
    
    # Generate points for Gaussian fit curve
    x = np.linspace(min(data_clean), max(data_clean), 100)
    y = norm.pdf(x, mu, std)
    
    # Add Gaussian fit
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        name=f'Gaussian fit (μ={mu:.3f}, σ={std:.3f})',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Log Variation",
        yaxis_title="Density", 
        showlegend=True
    )
    
    fig.show()



# -

plot_hist_with_gaussian(df_30s["tick_variation"], "Histogram of Log Variations - 20 Seconds Sampling")
plot_hist_with_gaussian(df_1min["tick_variation"], "Histogram of Log Variations - 1 Minute Sampling")
plot_hist_with_gaussian(df_5min["tick_variation"], "Histogram of Log Variations - 5 Minutes Sampling")



