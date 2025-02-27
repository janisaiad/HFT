import dotenv
import os
import polars as pl
import numpy as np
from datetime import datetime
from tqdm import tqdm
import json

dotenv.load_dotenv()

FOLDER_PATH = os.getenv("FOLDER_PATH") if os.getenv("FOLDER_PATH") else "/home/janis/EAP1/HFT_QR_RL/data/smash4/DB_MBP_10/"

# Fonction pour calculer la taille d'un dossier
def get_folder_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    return total_size

# Trier les stocks par taille de dossier
list_of_stocks = [(stock, get_folder_size(os.path.join(FOLDER_PATH, stock))) 
                  for stock in os.listdir(FOLDER_PATH)]
list_of_stocks.sort(key=lambda x: x[1])  # Tri par taille croissante
list_of_stocks = [stock for stock, _ in list_of_stocks]

# Création du dossier pour les résultats JSON
json_output_dir = "results/summarystats"
os.makedirs(json_output_dir, exist_ok=True)

# Création du DataFrame pour stocker les statistiques
columns = [
    "TICKER", "Tick_size", "Min_price", "Max_price", "Mean_trades_per_day",
    "Average_spread", "Max_spread", "Mean_volume_per_trade", "Median_volume_per_trade",
    "StdDev_volume_per_trade", "Transactions_at_bid_pct", "Average_duration_between_moves",
    "Median_duration_between_moves", "Max_duration_between_moves", "StdDev_duration_between_moves",
    "Number_of_jumps_week", "Average_jump_size", "Min_jump_size", "Max_jump_size",
    "StdDev_jump_size", "Prop_jumps_size_1", "Prop_jumps_size_minus1",
    "Prop_jumps_size_2", "Prop_jumps_size_minus2", "Prop_jumps_size_3",
    "Prop_jumps_size_minus3"
]

df_stats = pl.DataFrame(schema={col: pl.Float64 for col in columns})

# Constante pour la conversion des prix (1e-9 selon le format)
PRICE_SCALE = 1e-9

for stock in tqdm(list_of_stocks, desc="Processing stocks"):
    stock_path = os.path.join(FOLDER_PATH, stock)
    list_of_files = os.listdir(stock_path)
    
    # Filtrer pour ne garder que les fichiers parquet
    parquet_files = [f for f in list_of_files if f.endswith('.parquet')]
    
    all_data = []
    for file in tqdm(parquet_files, desc=f"Processing {stock} files", leave=False):
        file_path = os.path.join(stock_path, file)
        df = pl.read_parquet(file_path)
        all_data.append(df)
    
    if not all_data:
        continue
        
    # Concaténer tous les fichiers pour ce stock
    stock_data = pl.concat(all_data)
    
    # Convertir les prix en utilisant le facteur d'échelle
    stock_data = stock_data.with_columns([
        (pl.col("price") * PRICE_SCALE).alias("price_scaled"),
        (pl.col("bid_px_00") * PRICE_SCALE).alias("bid_price_scaled"),
        (pl.col("ask_px_00") * PRICE_SCALE).alias("ask_price_scaled")
    ])
    
    # Calcul des statistiques
    stats = {
        "TICKER": stock,
        "Tick_size": float(stock_data["price_scaled"].diff().abs().min() or 0),
        "Min_price": float(stock_data["price_scaled"].min()),
        "Max_price": float(stock_data["price_scaled"].max()),
        "Mean_trades_per_day": float(stock_data.filter(pl.col("action") == "T").groupby(
            pl.col("ts_event").cast(pl.Datetime).dt.date()
        ).count().mean()["count"]),
        "Average_spread": float((stock_data["ask_price_scaled"] - stock_data["bid_price_scaled"]).mean()),
        "Max_spread": float((stock_data["ask_price_scaled"] - stock_data["bid_price_scaled"]).max()),
        "Mean_volume_per_trade": float(stock_data.filter(pl.col("action") == "T")["size"].mean()),
        "Median_volume_per_trade": float(stock_data.filter(pl.col("action") == "T")["size"].median()),
        "StdDev_volume_per_trade": float(stock_data.filter(pl.col("action") == "T")["size"].std()),
        "Transactions_at_bid_pct": float(
            stock_data.filter(pl.col("action") == "T")
            .with_column(pl.col("price_scaled") == pl.col("bid_price_scaled"))
            .select(pl.col("price_scaled") == pl.col("bid_price_scaled"))
            .mean() * 100
        ),
    }
    
    # Calcul des durées entre mouvements (en nanosecondes)
    time_diffs = stock_data.filter(pl.col("price_scaled").diff() != 0)["ts_event"].diff()
    stats.update({
        "Average_duration_between_moves": float(time_diffs.mean()),
        "Median_duration_between_moves": float(time_diffs.median()),
        "Max_duration_between_moves": float(time_diffs.max()),
        "StdDev_duration_between_moves": float(time_diffs.std()),
    })
    
    # Calcul des sauts de prix
    price_jumps = stock_data["price_scaled"].diff()
    jumps = price_jumps[price_jumps != 0]
    
    stats.update({
        "Number_of_jumps_week": int(len(jumps)),
        "Average_jump_size": float(jumps.mean() * 1e5),  # × 10^-5 comme demandé
        "Min_jump_size": float(jumps.min()),
        "Max_jump_size": float(jumps.max()),
        "StdDev_jump_size": float(jumps.std()),
        "Prop_jumps_size_1": float((jumps == PRICE_SCALE).mean()),
        "Prop_jumps_size_minus1": float((jumps == -PRICE_SCALE).mean()),
        "Prop_jumps_size_2": float((jumps == 2 * PRICE_SCALE).mean()),
        "Prop_jumps_size_minus2": float((jumps == -2 * PRICE_SCALE).mean()),
        "Prop_jumps_size_3": float((jumps == 3 * PRICE_SCALE).mean()),
        "Prop_jumps_size_minus3": float((jumps == -3 * PRICE_SCALE).mean()),
    })
    
    # Sauvegarder les statistiques dans un fichier JSON pour ce stock
    json_file_path = os.path.join(json_output_dir, f"{stock}_stats.json")
    with open(json_file_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    # Ajouter les statistiques au DataFrame principal
    df_stats = df_stats.vstack(pl.DataFrame([stats]))

# Sauvegarder les résultats complets en parquet
output_path = "results/summary_statistics.parquet"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_stats.write_parquet(output_path)

# Afficher un aperçu des résultats
print(df_stats)

