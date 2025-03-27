from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple
import os
import polars as pl
import dotenv
from tqdm import tqdm
import plotly.graph_objects as go


FOLDER_PATH = os.getenv("FOLDER_PATH")
print(FOLDER_PATH)

dotenv.load_dotenv()


LIST_STOCKS_SIZE = {
    "GOOGL": "5.6G", "AAPL": "3.1G", "AMZN": "3.1G", "AAL": "2.2G", "MSFT": "2.1G",
    "GT": "2.1G", "INTC": "1.5G", "IOVA": "1.5G", "PTEN": "1.5G", "MLCO": "1.4G",
    "PTON": "1.4G", "VLY": "1.1G", "VOD": "951M", "CSX": "619M", "WB": "591M",
    "BGC": "591M", "GRAB": "454M", "KHC": "428M", "HLMN": "390M", "IEP": "342M",
    "GBDC": "338M", "WBD": "327M", "PSNY": "312M", "NTAP": "228M", "GEO": "199M",
    "LCID": "163M", "GCMG": "160M", "CXW": "108M", "RIOT": "36M", "HL": "18M",
    "CX": "4.8M", "ERIC": "4.2M", "UA": "2.7M"
}

for stock in tqdm(LIST_STOCKS_SIZE, desc="Processing stocks"):
    for file in tqdm(os.listdir(f"{FOLDER_PATH}{stock}"), desc=f"Processing {stock}"):
        try:
            df =  df = pl.read_parquet(f"{FOLDER_PATH}{stock}/{file}")


            num_entries_by_publisher = df.group_by("publisher_id").len().sort("len", descending=True)
            if len(num_entries_by_publisher) > 1:
                df = df.filter(pl.col("publisher_id") == 41)
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
            os.makedirs(f"{FOLDER_PATH}/data/hawkes_dataset/{stock}", exist_ok=True)
            hawkes_process.write_parquet(f"{FOLDER_PATH}/data/hawkes_dataset/{stock}/{file}")
        except Exception as e:
            print(f"Error processing {stock}/{file}: {e}")
            pass




