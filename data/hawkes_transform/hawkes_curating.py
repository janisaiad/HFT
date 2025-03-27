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
# -

df =  df = pl.read_parquet(f"{FOLDER_PATH}INTC/INTC_2024-07-22.parquet")

df.head(20)

num_entries_by_publisher = df.group_by("publisher_id").len().sort("len", descending=True)
if len(num_entries_by_publisher) > 1:
    df = df.filter(pl.col("publisher_id") == 41)

print(num_entries_by_publisher)

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
    y=mid_price,
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



# +
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple

# Définir les types d'événements
class EventType(Enum):
    PRICE_UP = "PRICE_UP"       # P(a)
    PRICE_DOWN = "PRICE_DOWN"   # P(b)
    TRADE_ASK = "TRADE_ASK"     # T(a)
    TRADE_BID = "TRADE_BID"     # T(b)
    LIMIT_ASK = "LIMIT_ASK"     # L(a)
    LIMIT_BID = "LIMIT_BID"     # L(b)
    CANCEL_ASK = "CANCEL_ASK"   # C(a)
    CANCEL_BID = "CANCEL_BID"   # C(b)



# +
def create_hawkes_counting_process(df: pl.DataFrame) -> pl.DataFrame:
    # Traiter les valeurs nulles dans bid_px_00 et ask_px_00
    df = df.with_columns([
        pl.when(pl.col("bid_px_00").is_null() | pl.col("ask_px_00").is_null())
        .then(None)
        .otherwise((pl.col("bid_px_00") + pl.col("ask_px_00")) / 2)
        .alias("mid_price")
    ])
    
    # Calculer les changements de prix
    df = df.with_columns([
        pl.col("mid_price").diff().alias("price_change")
    ])
    
    # Créer les indicateurs (traiter les valeurs nulles)
    df = df.with_columns([
        # Prix
        pl.when(pl.col("price_change") > 0).then(1).otherwise(0).alias("P_a"),
        pl.when(pl.col("price_change") < 0).then(1).otherwise(0).alias("P_b"),
        
        # Trades
        pl.when((pl.col("action") == "T") & (pl.col("side") == "A") & 
                (pl.col("price_change").is_null() | (pl.col("price_change") == 0)))
        .then(1).otherwise(0).alias("T_a"),
        
        pl.when((pl.col("action") == "T") & (pl.col("side") == "B") & 
                (pl.col("price_change").is_null() | (pl.col("price_change") == 0)))
        .then(1).otherwise(0).alias("T_b"),
        
        # Limit orders
        pl.when((pl.col("action") == "A") & (pl.col("side") == "A") & 
                (pl.col("price_change").is_null() | (pl.col("price_change") == 0)))
        .then(1).otherwise(0).alias("L_a"),
        
        pl.when((pl.col("action") == "A") & (pl.col("side") == "B") & 
                (pl.col("price_change").is_null() | (pl.col("price_change") == 0)))
        .then(1).otherwise(0).alias("L_b"),
        
        # Cancel orders
        pl.when((pl.col("action") == "C") & (pl.col("side") == "A") & 
                (pl.col("price_change").is_null() | (pl.col("price_change") == 0)))
        .then(1).otherwise(0).alias("C_a"),
        
        pl.when((pl.col("action") == "C") & (pl.col("side") == "B") & 
                (pl.col("price_change").is_null() | (pl.col("price_change") == 0)))
        .then(1).otherwise(0).alias("C_b")
    ])
    
    # Créer les processus cumulatifs en utilisant cum_sum()
    counting_process = df.select([
        "ts_event",
        pl.col("P_a").cum_sum().alias("P_a"),
        pl.col("P_b").cum_sum().alias("P_b"),
        pl.col("T_a").cum_sum().alias("T_a"),
        pl.col("T_b").cum_sum().alias("T_b"),
        pl.col("L_a").cum_sum().alias("L_a"),
        pl.col("L_b").cum_sum().alias("L_b"),
        pl.col("C_a").cum_sum().alias("C_a"),
        pl.col("C_b").cum_sum().alias("C_b")
    ])
    
    return counting_process

# Usage
hawkes_process = create_hawkes_counting_process(df)
print(hawkes_process)

# Pour sauvegarder en parquet
# hawkes_process.write_parquet("hawkes_counting_process.parquet")
# -




