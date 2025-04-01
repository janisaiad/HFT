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

import polars as pl
import plotly.graph_objects as go
import os

# +

# day_list = ["2024-08-05","2024-08-06"]#,"2024-08-07"] # ,"2023-07-20","2023-07-21", "2023-07-24", "2023-07-25", "2023-07-26", "2023-07-27", "2023-07-28"]
stock = "YMM"
day_list = sorted(os.listdir(f"/home/janis/EAP1/HFT_QR_RL/data/smash4/DB_MBP_10/{stock}/"))[:2]
print(day_list)
# -

df = pl.concat([
    df for df in [
        pl.read_parquet(f"/home/janis/EAP1/HFT_QR_RL/data/smash4/DB_MBP_10/{stock}/{day}")
        for day in day_list
    ] if df.width >=10
])

# Tri heure

start_hour = 9
end_hour = 16

df = df.filter((pl.col("ts_event").dt.hour() >= start_hour) & 
               (pl.col("ts_event").dt.hour() < end_hour) &
               ((pl.col("ts_event").dt.hour() != start_hour) | (pl.col("ts_event").dt.minute() >= 30)))


# > A NE PAS RUN

df = df.filter(
    (pl.col("ts_event").dt.date().cast(pl.Utf8).is_in(day_list[:-1]) & (pl.col("ts_event").dt.hour() == end_hour)) |
    (pl.col("ts_event").dt.date().cast(pl.Utf8).is_in(day_list[1:]) & (pl.col("ts_event").dt.hour() == start_hour))
)

num_entries_by_publisher = df.group_by("publisher_id").len().sort("len", descending=True)
num_entries_by_publisher.head(10)

# +

df= df.filter(pl.col("publisher_id") == 41)

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


# Update layout
fig.update_layout(
    title='Order Book and bid/ask',
    xaxis_title='Time',
    yaxis_title='Price',
    showlegend=True
)

fig.show()


# -










