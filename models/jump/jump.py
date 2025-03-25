
# Objectif, coder le jump-score Jt = rt / (sigma_t * ft)

# IMPORTS
import os
import polars as pl
import numpy as np

from tqdm import tqdm

from curating.mid_price import curate_mid_price

folder_path = "/Users/fwr/Desktop/DB_MBP_10/"

# 0 - Charger
# !!!! changer le folder path
# commencer avec un stock peu tradé
# choisir un rolling time (dans le papier = 1 minute)

stock = "INTC" # exemple
file = "INTC_2024-07-25.parquet"

tabl = curate_mid_price(stock, file, folder_path=folder_path)
tabl = tabl.select(["ts_event", "mid_price"])
# print(tabl.shape)
# print(tabl.columns)

# les deux colonnes intéressantes : publisher_id et midprice



# 1 _ obtenir rt


# -> rt : 1 minute return time series
# formule : rt = log(mt/m_t-1)


def find_return_time_series(stock, file, folder_path = folder_path):
    
    df = curate_mid_price(stock, file, folder_path=folder_path)
    df = df.select(["ts_event", "mid_price"])
    
    df_shifted = df.with_columns((pl.col("ts_event") + pl.duration(minutes=-1)).alias("ts_minus_1min").cast(pl.Datetime("ns", time_zone="US/Eastern")))
    df_shifted = df_shifted.select(["mid_price", "ts_minus_1min"])
    # print(df[10,0])
    # print(df_shifted[10, 1])
    # print(df_shifted.columns)
    # print(df.dtypes)
    # print(df_shifted.dtypes)
    
    df_joined = df_shifted.join_asof(df, left_on="ts_minus_1min", right_on="ts_event", strategy="backward", suffix="_prev")

    # print(df_joined.columns)

    df_final = df_joined.with_columns((pl.col("mid_price") / pl.col("mid_price_prev")).log().alias("rt")).select(["ts_event", "mid_price", "rt"])

    # print(df_final.columns)
    
    return df_final

final = find_return_time_series(stock, file, folder_path=folder_path)



# 2 _ obtenir sigma_t

# estimateur de la volatilité locale

K = 390

def find_local_volatility(df_final, K):

    factor = np.pi / (2 * K)

    df_sigma = df_final.with_columns(((pl.col("rt").abs() * pl.col("rt").abs().shift(1)).rolling_sum(window_size=K, min_periods=K)* factor).sqrt().alias("sigma_t"))
    print(df_sigma.columns)

    return df_sigma

sigma = find_local_volatility(final, K)

# 3 _ obtenir ft

# estimateur de la composante périodique intra-journalière (voir appendice)

def find_periodicity_component(df_final):
    
    # Étape 1: Calcul de r_i chapeau
    df_final = df_final.with_columns(
        (pl.col("rt") / pl.col("sigma_t2").sqrt()).alias("r_hat")
    )

    # Étape 2: Trouver les r_j qui sont dans les 1 minute suivantes
    df_shifted = df_final.with_columns(
        (pl.col("ts_event") + pl.duration(minutes=1)).alias("ts_plus_1min")
    )

    df_future = df_final.join_asof(
        df_shifted, left_on="ts_event", right_on="ts_plus_1min",
        strategy="forward", suffix="_future"
    )

    # Étape 3: Calcul de Wi
    df_future = df_future.with_columns([
        ((-pl.col("r_hat_future").pow(2) + x) > 0).cast(pl.Float64).alias("Theta"),
        pl.col("r_hat_future").pow(2).alias("r_hat2")
    ])

    Wi_numerator = (1.081 * (df_future["Theta"] * df_future["r_hat2"]).sum()).sqrt()
    Wi_denominator = (df_future["Theta"].sum()).sqrt()

    df_future = df_future.with_columns(
        (Wi_numerator / Wi_denominator).alias("Wi")
    )

    # Étape 4: Normalisation pour obtenir f_i
    df_future = df_future.with_columns(
        (pl.col("Wi") / (pl.col("Wi").pow(2).rolling_mean(window_size=len(df_future)).sqrt())).alias("ft")
    )

    print(df_future.select(["ts_event", "ft"]))


