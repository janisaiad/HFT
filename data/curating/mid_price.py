import os
import polars as pl

from tqdm import tqdm

folder_path = "/home/janis/EAP1/HFT_QR_RL/data/smash4/DB_MBP_10/"

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
    
    
    
    
    