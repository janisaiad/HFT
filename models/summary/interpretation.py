import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

# Configuration de style pour les plots
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 300

# Création des dossiers pour les résultats
RESULTS_DIR = Path("/home/janis/HFTP2/HFT/results/summarystats")
PLOTS_DIR = RESULTS_DIR / "plots/"
PLOTS_DIR.mkdir(exist_ok=True)

def calculate_optimal_bins(data):
    """Calcule le nombre optimal de bins pour l'histogramme"""
    # Règle de Freedman-Diaconis
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(data) ** (1/3))
    if bin_width == 0:
        return min(50, len(np.unique(data)))
    return int(np.ceil((data.max() - data.min()) / bin_width))

def add_distribution_stats(ax, data, title):
    """Ajoute les statistiques descriptives au plot"""
    stats_text = f'n = {len(data)}\n'
    stats_text += f'μ = {np.mean(data):.2f}\n'
    stats_text += f'σ = {np.std(data):.2f}\n'
    stats_text += f'med = {np.median(data):.2f}\n'
    
    # Test de normalité
    _, p_value = stats.normaltest(data)
    stats_text += f'p-norm = {p_value:.2e}'
    
    # Ajouter le texte dans un cadre
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.95, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=props,
            fontsize=10)

def create_distribution_plots(df):
    """Crée des histogrammes intelligents pour chaque statistique"""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        data = df[col].dropna()
        if len(data) == 0:
            continue
            
        # Créer deux subplots côte à côte
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Premier plot: Distribution normale
        n_bins = calculate_optimal_bins(data)
        sns.histplot(data=data, bins=n_bins, kde=True, ax=ax1)
        ax1.set_title(f'Distribution de {col}')
        add_distribution_stats(ax1, data, col)
        
        # Deuxième plot: Échelle logarithmique si pertinent
        if data.min() > 0 and data.max() / data.min() > 10:
            sns.histplot(data=data, bins=n_bins, kde=True, ax=ax2)
            ax2.set_yscale('log')
            if data.min() > 0:
                ax2.set_xscale('log')
            ax2.set_title(f'Distribution de {col} (échelle log)')
        else:
            # Si log n'est pas pertinent, montrer un boxplot
            sns.boxplot(data=data, ax=ax2)
            ax2.set_title(f'Boxplot de {col}')
        
        # Ajouter des statistiques sur les outliers
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        outliers = data[(data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)]
        if len(outliers) > 0:
            outlier_text = f'Outliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)\n'
            outlier_text += f'Min: {outliers.min():.2f}\n'
            outlier_text += f'Max: {outliers.max():.2f}'
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax2.text(0.95, 0.05, outlier_text,
                    transform=ax2.transAxes,
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=props,
                    fontsize=10)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f'dist_{col}.png')
        plt.close()

def load_all_stats():
    """Charge tous les fichiers JSON de statistiques"""
    all_data = []
    for file in RESULTS_DIR.glob("*_stats.json"):
        if file.is_file():
            with open(file, 'r') as f:
                data = json.load(f)
                # Aplatir la structure pour faciliter l'analyse
                ticker = data.pop("TICKER")
                if isinstance(ticker, dict):
                    flat_data = ticker
                else:
                    flat_data = {"stock": ticker}
                flat_data.update(data)
                all_data.append(flat_data)
    return pd.DataFrame(all_data)

def create_correlation_matrix(df):
    """Crée une heatmap de corrélation"""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Matrice de Corrélation')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'correlation_matrix.png')
    plt.close()

def create_summary_stats(df):
    """Calcule les statistiques descriptives pour chaque métrique"""
    stats = df.describe()
    stats_dict = stats.to_dict()
    
    # Sauvegarder en JSON
    with open(RESULTS_DIR / 'summary_statistics.json', 'w') as f:
        json.dump(stats_dict, f, indent=4)
    
    return stats_dict

def create_boxplots(df):
    """Crée des boxplots pour visualiser la distribution des métriques"""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Diviser les colonnes en groupes de 3 pour une meilleure visualisation
    n_cols = 3
    n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_columns):
        sns.boxplot(data=df, y=col, ax=axes[idx])
        axes[idx].set_title(col)
        axes[idx].tick_params(axis='x', rotation=45)
    
    # Supprimer les subplots vides
    for idx in range(len(numeric_columns), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'boxplots.png')
    plt.close()

def create_scatter_matrix(df):
    """Crée une matrice de scatter plots pour les métriques principales"""
    main_metrics = [
        'Mean number of trades per day',
        'Average spread',
        'Mean volume per trade',
        'Average duration between moves',
        'Number of jumps over the week'
    ]
    
    if all(metric in df.columns for metric in main_metrics):
        sns.pairplot(df[main_metrics])
        plt.savefig(PLOTS_DIR / 'scatter_matrix.png')
        plt.close()

def main():
    print("Chargement des données...")
    df = load_all_stats()
    
    print("Création des visualisations...")
    create_distribution_plots(df)
    create_correlation_matrix(df)
    create_boxplots(df)
    create_scatter_matrix(df)
    
    print("Calcul des statistiques descriptives...")
    summary_stats = create_summary_stats(df)
    
    print(f"Analyse terminée. Les résultats sont disponibles dans {RESULTS_DIR}")
    print(f"Les visualisations sont disponibles dans {PLOTS_DIR}")

if __name__ == "__main__":
    main()
