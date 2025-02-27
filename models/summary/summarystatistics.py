import dotenv
import os
import polars as pl
import numpy as np
from datetime import datetime, time
from tqdm import tqdm
import json
import gc

dotenv.load_dotenv()

FOLDER_PATH = os.getenv("FOLDER_PATH") if os.getenv("FOLDER_PATH") else "/home/janis/3A/EA/HFT_QR_RL/data/smash4/DB_MBP_10/"

# Colonnes requises pour l'analyse
REQUIRED_COLUMNS = ["price", "bid_px_00", "ask_px_00", "size", "action", "ts_event"]

# Création du dossier pour les résultats JSON et les logs
json_output_dir = "results/summarystats"
os.makedirs(json_output_dir, exist_ok=True)

# Fichier de log pour les problèmes
log_file = os.path.join(json_output_dir, "processing_issues.txt")

def log_issue(message):
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now()}: {message}\n")

def get_folder_size(path):
    """Calculer la taille totale d'un dossier"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)
    except Exception as e:
        print(f"Erreur lors du calcul de la taille pour {path}: {e}")
        return 0
    return total_size

class StockStats:
    def __init__(self):
        self.min_price = float('inf')
        self.max_price = float('-inf')
        self.sum_price = 0
        self.count_price = 0
        self.total_trades = 0
        self.dates = set()
        self.total_rows = 0
        self.min_date = None
        self.max_date = None
    
    def update(self, df):
        # Mise à jour des statistiques de prix
        prices = df["price_scaled"].to_numpy()
        self.min_price = min(self.min_price, float(np.min(prices)))
        self.max_price = max(self.max_price, float(np.max(prices)))
        self.sum_price += float(np.sum(prices))
        self.count_price += len(prices)
        
        # Mise à jour des trades
        self.total_trades += df.filter(pl.col("action") == "T").height
        
        # Mise à jour des dates
        dates = df["datetime"].dt.date().unique()
        self.dates.update(dates.to_list())
        
        # Mise à jour du nombre total de lignes
        self.total_rows += len(df)
        
        # Mise à jour des dates min/max
        df_min_date = df["datetime"].min()
        df_max_date = df["datetime"].max()
        if self.min_date is None or df_min_date < self.min_date:
            self.min_date = df_min_date
        if self.max_date is None or df_max_date > self.max_date:
            self.max_date = df_max_date
    
    def get_stats(self):
        return {
            "data_shape": {"rows": self.total_rows},
            "date_range": {
                "start": self.min_date.strftime("%Y-%m-%d %H:%M:%S"),
                "end": self.max_date.strftime("%Y-%m-%d %H:%M:%S")
            },
            "basic_stats": {
                "min_price": self.min_price,
                "max_price": self.max_price,
                "mean_price": self.sum_price / self.count_price if self.count_price > 0 else 0,
                "total_trades": self.total_trades,
                "unique_days": len(self.dates)
            }
        }

# Constante pour la conversion des prix (1e-9 selon le format)
PRICE_SCALE = 1e-9

# Liste et trie les stocks par taille décroissante
print("Calcul de la taille des dossiers...")
stocks_with_size = []
for stock in os.listdir(FOLDER_PATH):
    stock_path = os.path.join(FOLDER_PATH, stock)
    if os.path.isdir(stock_path):
        size = get_folder_size(stock_path)
        stocks_with_size.append((stock, size))

# Tri par taille décroissante
stocks_with_size.sort(key=lambda x: x[1], reverse=True)
stocks = [stock for stock, _ in stocks_with_size]

print(f"Nombre total de stocks trouvés: {len(stocks)}")
print("Top 5 plus gros dossiers:")
for stock, size in stocks_with_size[:5]:
    print(f"{stock}: {size / (1024*1024*1024):.2f} GB")

for stock in tqdm(stocks, desc="Processing stocks"):
    stock_path = os.path.join(FOLDER_PATH, stock)
    
    try:
        # Liste tous les fichiers parquet pour ce stock
        parquet_files = [f for f in os.listdir(stock_path) if f.endswith('.parquet')]
        
        if not parquet_files:
            log_issue(f"{stock}: Aucun fichier parquet trouvé")
            continue
            
        # Essayer de lire le premier fichier pour vérifier la structure
        first_file = os.path.join(stock_path, parquet_files[0])
        try:
            test_df = pl.read_parquet(first_file)
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in test_df.columns]
            if missing_cols:
                log_issue(f"{stock}: Colonnes manquantes dans les données: {missing_cols}")
                log_issue(f"{stock}: Colonnes disponibles: {test_df.columns}")
                continue
            del test_df
        except Exception as e:
            log_issue(f"{stock}: Erreur lors de la lecture du premier fichier: {str(e)}")
            continue
            
        # Si on arrive ici, la structure est OK, on peut traiter tous les fichiers
        stats_accumulator = StockStats()
        
        for file in tqdm(parquet_files, desc=f"Processing {stock} files", leave=False):
            file_path = os.path.join(stock_path, file)
            try:
                # Lire et traiter le fichier
                df = pl.read_parquet(file_path, columns=REQUIRED_COLUMNS)
                
                # Filtrer les heures de trading (13h30-20h)
                df = (df
                     .with_columns([
                         pl.col("ts_event").cast(pl.Datetime).alias("datetime"),
                         (pl.col("price") * PRICE_SCALE).alias("price_scaled")
                     ])
                     .filter(
                         (pl.col("datetime").dt.hour() >= 13) | 
                         ((pl.col("datetime").dt.hour() == 13) & (pl.col("datetime").dt.minute() >= 30))
                     )
                     .filter(
                         pl.col("datetime").dt.hour() < 20
                     ))
                
                if len(df) > 0:
                    stats_accumulator.update(df)
                
                del df
                gc.collect()
                    
            except Exception as e:
                log_issue(f"{stock}: Erreur lors du traitement de {file}: {str(e)}")
                continue
        
        if stats_accumulator.total_rows == 0:
            log_issue(f"{stock}: Aucune donnée valide après filtrage")
            continue
            
        # Obtenir les statistiques finales
        stats = {"TICKER": stock}
        stats.update(stats_accumulator.get_stats())
        
        # Sauvegarder immédiatement les stats pour ce stock
        json_file_path = os.path.join(json_output_dir, f"{stock}_stats.json")
        with open(json_file_path, 'w') as f:
            json.dump(stats, f, indent=4)
            
    except Exception as e:
        log_issue(f"{stock}: Erreur générale: {str(e)}")
        continue

print(f"\nTraitement terminé. Consultez {log_file} pour les détails des problèmes rencontrés.")

