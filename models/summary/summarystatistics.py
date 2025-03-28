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

# Nettoyage des fichiers JSON existants - Forcer la suppression
for file in os.listdir(json_output_dir):
    if file.endswith('.json'):
        file_path = os.path.join(json_output_dir, file)
        try:
            os.remove(file_path)
            print(f"Suppression: {file_path}")
        except Exception as e:
            print(f"Erreur lors de la suppression de {file_path}: {e}")

# Fichier de log pour les problèmes
log_file = os.path.join(json_output_dir, "processing_issues.txt")

# Supprimer le fichier de log s'il existe
if os.path.exists(log_file):
    os.remove(log_file)

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
        # Tick size est fixe à 0.01 comme demandé
        self.tick_size = 0.01
        
        # Stats de prix de base (basées sur mid price)
        self.min_price = float('inf')
        self.max_price = float('-inf')
        self.sum_price = 0
        self.count_price = 0
        
        # Stats de trades
        self.total_trades = 0
        self.trades_at_bid = 0
        self.sum_volume = 0
        self.volumes = []  # Pour calculer médiane et écart-type
        
        # Stats temporelles
        self.dates = set()
        self.total_rows = 0
        self.min_date = None
        self.max_date = None
        self.time_diffs = []  # Pour calculer médiane et écart-type des durées
        
        # Stats de spread
        self.sum_spread = 0
        self.count_spread = 0
        self.max_spread = float('-inf')
        
        # Stats de sauts
        self.jumps = []  # Pour calculer les stats sur les sauts
        self.jump_counts = {
            1: 0, -1: 0,
            2: 0, -2: 0,
            3: 0, -3: 0
        }
        self.total_jumps = 0
    
    def update(self, df):
        # Filtrer les lignes où bid et ask sont valides
        df_filtered = df.filter(
            (pl.col("bid_price_scaled").is_not_null()) & 
            (pl.col("ask_price_scaled").is_not_null()) &
            (pl.col("bid_price_scaled") > 0) & 
            (pl.col("ask_price_scaled") > 0) &
            (pl.col("ask_price_scaled") > pl.col("bid_price_scaled"))
        )
        
        if len(df_filtered) > 0:
            # Calculer le mid price
            df_valid = df_filtered.with_columns([
                ((pl.col("bid_price_scaled") + pl.col("ask_price_scaled")) / 2.0).alias("mid_price")
            ])
            
            # Mise à jour des statistiques de prix basées sur mid price
            mid_prices = df_valid["mid_price"].to_numpy()
            self.min_price = min(self.min_price, float(np.min(mid_prices)))
            self.max_price = max(self.max_price, float(np.max(mid_prices)))
            self.sum_price += float(np.sum(mid_prices))
            self.count_price += len(mid_prices)
            
            # Calcul des spreads en ticks (divisés par le tick size fixe)
            spreads = (df_valid["ask_price_scaled"] - df_valid["bid_price_scaled"]).to_numpy() / self.tick_size
            self.sum_spread += float(np.sum(spreads))
            self.count_spread += len(spreads)
            
            if len(spreads) > 0:
                self.max_spread = max(self.max_spread, float(np.max(spreads)))
        
            # Stats de trades (seulement pour les lignes avec mid price valide)
            trades = df_valid.filter(pl.col("action") == "T")
            self.total_trades += trades.height
            
            if trades.height > 0:
                # Volumes des trades
                trade_volumes = trades["size"].to_numpy()
                self.sum_volume += float(np.sum(trade_volumes))
                self.volumes.extend(trade_volumes.tolist())
                
                # Trades au bid (comparé avec le mid price)
                trades_at_bid = trades.filter(pl.col("price_scaled") <= pl.col("mid_price")).height
                self.trades_at_bid += trades_at_bid
            
            # Mise à jour des dates
            dates = df_valid["datetime"].dt.date().unique()
            self.dates.update(dates.to_list())
            
            # Mise à jour du nombre total de lignes
            self.total_rows += len(df_valid)
            
            # Mise à jour des dates min/max
            df_min_date = df_valid["datetime"].min()
            df_max_date = df_valid["datetime"].max()
            if self.min_date is None or df_min_date < self.min_date:
                self.min_date = df_min_date
            if self.max_date is None or df_max_date > self.max_date:
                self.max_date = df_max_date
                
            # Calcul des durées entre mouvements de mid price
            if len(mid_prices) > 1:
                price_changes = df_valid.filter(pl.col("mid_price").diff() != 0)
                if len(price_changes) > 1:
                    time_diffs = price_changes["ts_event"].diff().drop_nulls()
                    self.time_diffs.extend(time_diffs.to_numpy().tolist())
                
            # Calcul des sauts de prix basés sur mid price (en ticks)
            if len(mid_prices) > 1:
                price_diffs = df_valid["mid_price"].diff().drop_nulls() / self.tick_size
                non_zero_diffs = price_diffs[np.abs(price_diffs) >= 1]  # Ignorer les changements inférieurs à 1 tick
                
                if len(non_zero_diffs) > 0:
                    self.jumps.extend(non_zero_diffs.to_numpy().tolist())
                    
                    # Compter les sauts de différentes tailles
                    for jump_size in [1, -1, 2, -2, 3, -3]:
                        count = np.sum(np.isclose(non_zero_diffs, jump_size, atol=0.1))
                        self.jump_counts[jump_size] += count
                        self.total_jumps += count
    
    def get_stats(self):
        # Convertir les listes en arrays numpy pour les calculs
        time_diffs_arr = np.array(self.time_diffs) if self.time_diffs else np.array([0])
        volumes_arr = np.array(self.volumes) if self.volumes else np.array([0])
        jumps_arr = np.array(self.jumps) if self.jumps else np.array([0])
        
        # Informations de debug
        print(f"Statistiques pour le tick: {self.tick_size}")
        print(f"Min price: {self.min_price}")
        print(f"Max price: {self.max_price}")
        print(f"Average spread: {self.sum_spread / self.count_spread if self.count_spread > 0 else 0}")
        print(f"Trades at bid: {(self.trades_at_bid / self.total_trades * 100) if self.total_trades > 0 else 0}%")
        print(f"Nombre de sauts: {self.total_jumps}")
        
        # Calculer les statistiques dans le format demandé
        stats = {
            "TICKER": {
                "Tick size": float(self.tick_size),
                "Min price": self.min_price,
                "Max price": self.max_price,
                "Mean number of trades per day": self.total_trades / len(self.dates) if self.dates else 0,
                "Average spread (in ticks)": self.sum_spread / self.count_spread if self.count_spread > 0 else 0,
                "Max spread (in ticks)": self.max_spread
            },
            "Mean volume per trade": self.sum_volume / self.total_trades if self.total_trades > 0 else 0,
            "Average duration between moves": float(np.mean(time_diffs_arr)),
            "Median duration between moves": float(np.median(time_diffs_arr)),
            "Max duration between moves": float(np.max(time_diffs_arr)),
            "Stand. Dev. duration between moves": float(np.std(time_diffs_arr)),
            "Median volume per trade": float(np.median(volumes_arr)),
            "Stand. Dev. Volume per trade": float(np.std(volumes_arr)),
            "Transactions at the Bid price (%)": (self.trades_at_bid / self.total_trades * 100) if self.total_trades > 0 else 0,
            "Number of jumps over the week": self.total_jumps * 5 / len(self.dates) if self.dates else 0,
            "Average size of the jumps (in ticks)": float(np.mean(jumps_arr)),
            "Min. size of the jumps (in ticks)": float(np.min(jumps_arr)),
            "Max. size of the jumps (in ticks)": float(np.max(jumps_arr)),
            "Stand. Dev. size of the jumps (in ticks)": float(np.std(jumps_arr)),
            "Prop. jumps of size 1": self.jump_counts[1] / self.total_jumps if self.total_jumps > 0 else 0,
            "Prop. jumps of size -1": self.jump_counts[-1] / self.total_jumps if self.total_jumps > 0 else 0,
            "Prop. jumps of size 2": self.jump_counts[2] / self.total_jumps if self.total_jumps > 0 else 0,
            "Prop. jumps of size -2": self.jump_counts[-2] / self.total_jumps if self.total_jumps > 0 else 0,
            "Prop. jumps of size 3": self.jump_counts[3] / self.total_jumps if self.total_jumps > 0 else 0,
            "Prop. jumps of size -3": self.jump_counts[-3] / self.total_jumps if self.total_jumps > 0 else 0
        }
        
        return stats

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
    json_file_path = os.path.join(json_output_dir, f"{stock}_stats.json")
    
    # Vérifier si le fichier JSON existe déjà et le supprimer
    if os.path.exists(json_file_path):
        try:
            os.remove(json_file_path)
            print(f"Suppression existante: {json_file_path}")
        except Exception as e:
            log_issue(f"Erreur suppression: {json_file_path}: {e}")
    
    try:
        stock_path = os.path.join(FOLDER_PATH, stock)
        
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
                
                # Filtrer les heures de trading (9:30-16:00) et préparer toutes les colonnes
                df = (df
                     .with_columns([
                         pl.col("ts_event").cast(pl.Datetime).alias("datetime"),
                         pl.col("price").alias("price_scaled"),
                         pl.col("bid_px_00").alias("bid_price_scaled"),
                         pl.col("ask_px_00").alias("ask_price_scaled")
                     ])
                     .filter(
                         (pl.col("datetime").dt.hour() >= 10) | 
                         ((pl.col("datetime").dt.hour() == 9) & (pl.col("datetime").dt.minute() >= 30))
                     )
                     .filter(
                         pl.col("datetime").dt.hour() < 16
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
            
        # Obtenir les statistiques finales et sauvegarder
        stats = {"TICKER": stock}
        stats.update(stats_accumulator.get_stats())
        
        # Sauvegarder dans un nouveau fichier
        with open(json_file_path, 'w') as f:
            json.dump(stats, f, indent=4)
        
        # Vérifier que le fichier a bien été créé
        if os.path.exists(json_file_path):
            print(f"Fichier JSON créé avec succès: {json_file_path}")
        else:
            log_issue(f"{stock}: Échec de création du fichier JSON")
            
    except Exception as e:
        log_issue(f"{stock}: Erreur générale: {str(e)}")
        continue

print(f"\nTraitement terminé. Consultez {log_file} pour les détails des problèmes rencontrés.")

