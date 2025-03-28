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
stock = "UA"
# -

df = pl.read_parquet(f"{FOLDER_PATH}{stock}/{stock}_2024-07-22.parquet")


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

    return df


df = curate_mid_price(df,stock)

df.head(10)

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

df_cleaned.head(10)

# +
# Create different time-based resampled dataframes
df_1s = df_cleaned.group_by(pl.col("ts_event").dt.truncate("1s")).agg([
    pl.col("mid_price").last().alias("mid_price")
])

df_5s = df_cleaned.group_by(pl.col("ts_event").dt.truncate("5s")).agg([
    pl.col("mid_price").last().alias("mid_price")
])

df_20s = df_cleaned.group_by(pl.col("ts_event").dt.truncate("20s")).agg([
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

# Display the first few rows of each resampled dataframe
print("1 second sampling:")
print(df_1s.head())
print("\n5 seconds sampling:")
print(df_5s.head())
print("\n20 seconds sampling:")
print(df_20s.head())
print("\n1 minute sampling:")
print(df_1min.head())
print("\n5 minutes sampling:")
print(df_5min.head())
print("\n10 minutes sampling:")
print(df_10min.head())

# sorting by ts_event
df_1s = df_1s.sort("ts_event")
df_5s = df_5s.sort("ts_event")
df_20s = df_20s.sort("ts_event")
df_1min = df_1min.sort("ts_event")
df_5min = df_5min.sort("ts_event")



df_1s = df_1s.with_columns(tick_variation=pl.col("mid_price").diff())
df_1s = df_1s.with_columns(log_variation=pl.col("mid_price").log().diff())

df_5s = df_5s.with_columns(tick_variation=pl.col("mid_price").diff())
df_5s = df_5s.with_columns(log_variation=pl.col("mid_price").log().diff())

df_20s = df_20s.with_columns(tick_variation=pl.col("mid_price").diff())
df_20s = df_20s.with_columns(log_variation=pl.col("mid_price").log().diff())

df_1min = df_1min.with_columns(tick_variation=pl.col("mid_price").diff())
df_1min = df_1min.with_columns(log_variation=pl.col("mid_price").log().diff())

df_5min = df_5min.with_columns(tick_variation=pl.col("mid_price").diff())
df_5min = df_5min.with_columns(log_variation=pl.col("mid_price").log().diff())



# +
import plotly.graph_objects as go

# 1 Second sampling plot
fig1 = go.Figure()
fig1.add_trace(
    go.Scatter(x=df_1s["ts_event"], y=df_1s["mid_price"], name="Mid Price")
)
fig1.update_layout(
    title="1 Second Sampling",
    xaxis_title="Time",
    yaxis_title="Mid Price"
)
fig1.show()

# 5 Seconds sampling plot
fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(x=df_5s["ts_event"], y=df_5s["mid_price"], name="Mid Price")
)
fig2.update_layout(
    title="5 Seconds Sampling",
    xaxis_title="Time", 
    yaxis_title="Mid Price"
)
fig2.show()

# 20 Seconds sampling plot
fig3 = go.Figure()
fig3.add_trace(
    go.Scatter(x=df_20s["ts_event"], y=df_20s["mid_price"], name="Mid Price")
)
fig3.update_layout(
    title="20 Seconds Sampling",
    xaxis_title="Time",
    yaxis_title="Mid Price"
)
fig3.show()

# 1 Minute sampling plot
fig4 = go.Figure()
fig4.add_trace(
    go.Scatter(x=df_1min["ts_event"], y=df_1min["mid_price"], name="Mid Price")
)
fig4.update_layout(
    title="1 Minute Sampling",
    xaxis_title="Time",
    yaxis_title="Mid Price"
)
fig4.show()

# 5 Minutes sampling plot
fig5 = go.Figure()
fig5.add_trace(
    go.Scatter(x=df_5min["ts_event"], y=df_5min["mid_price"], name="Mid Price")
)
fig5.update_layout(
    title="5 Minutes Sampling", 
    xaxis_title="Time",
    yaxis_title="Mid Price"
)
fig5.show()



# +
import plotly.graph_objects as go

# 1 Second sampling plot
fig1 = go.Figure()
fig1.add_trace(
    go.Scatter(x=df_1s["ts_event"], y=df_1s["tick_variation"], name = "Tick variation")
)
fig1.update_layout(
    title="1 Second Sampling",
    xaxis_title="Time",
    yaxis_title="Mid Price"
)
fig1.show()

# 5 Seconds sampling plot
fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(x=df_5s["ts_event"], y=df_5s["tick_variation"], name = "Tick variation")
)
fig2.update_layout(
    title="5 Seconds Sampling",
    xaxis_title="Time", 
    yaxis_title="Mid Price"
)
fig2.show()

# 20 Seconds sampling plot
fig3 = go.Figure()
fig3.add_trace(
    go.Scatter(x=df_20s["ts_event"], y=df_20s["tick_variation"], name = "Tick variation")
)
fig3.update_layout(
    title="20 Seconds Sampling",
    xaxis_title="Time",
    yaxis_title="Mid Price"
)
fig3.show()

# 1 Minute sampling plot
fig4 = go.Figure()
fig4.add_trace(
    go.Scatter(x=df_1min["ts_event"], y=df_1min["tick_variation"], name = "Tick variation")
)
fig4.update_layout(
    title="1 Minute Sampling",
    xaxis_title="Time",
    yaxis_title="Mid Price"
)
fig4.show()

# 5 Minutes sampling plot
fig5 = go.Figure()
fig5.add_trace(
    go.Scatter(x=df_5min["ts_event"], y=df_5min["tick_variation"], name = "Tick variation")
)
fig5.update_layout(
    title="5 Minutes Sampling", 
    xaxis_title="Time",
    yaxis_title="Mid Price"
)
fig5.show()


# -






