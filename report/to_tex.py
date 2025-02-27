import json
import pandas as pd
from pathlib import Path

def format_number(x):
    """Formate les nombres pour LaTeX"""
    if isinstance(x, (int, float)):
        if abs(x) < 0.01 or abs(x) > 1000:
            return f"{x:.2e}"
        return f"{x:.2f}"
    return str(x)

def create_latex_table(df):
    """Crée une table LaTeX simple"""
    # Formater les données
    df = df.applymap(format_number)
    
    # Créer la table
    latex = [
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{l" + "r" * (len(df.columns)-1) + "}",
        "\\toprule"
    ]
    
    # En-têtes
    latex.append(" & ".join(df.columns) + " \\\\")
    latex.append("\\midrule")
    
    # Données
    for _, row in df.iterrows():
        latex.append(" & ".join(row.astype(str)) + " \\\\")
    
    # Fin
    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Statistiques par stock}",
        "\\end{table}"
    ])
    
    return "\n".join(latex)

def main():
    # Charger les données
    json_dir = Path("/home/janis/HFTP2/HFT/results/summarystats")
    all_data = []
    
    for file in json_dir.glob("*_stats.json"):
        if file.is_file() and not file.name == "summary_statistics.json":
            ticker = file.stem.replace("_stats", "")  # Extraire le ticker du nom du fichier
            with open(file, 'r') as f:
                data = json.load(f)
                stats = data["TICKER"]  # Prendre les stats de la section TICKER
                stats["Ticker"] = ticker  # Ajouter le ticker
                all_data.append(stats)
    
    # Créer DataFrame
    df = pd.DataFrame(all_data)
    
    # Colonnes importantes
    cols = [
        "Ticker",
        "Tick size",
        "Min price",
        "Max price",
        "Mean number of trades per day",
        "Average spread",
        "Max spread"
    ]
    
    # Sélectionner colonnes existantes
    cols = [col for col in cols if col in df.columns]
    df_selected = df[cols]
    
    # Générer et sauvegarder la table
    latex_table = create_latex_table(df_selected)
    output_file = Path("/home/janis/HFTP2/HFT/report/table_stats.tex")
    
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"Table LaTeX générée dans {output_file}")

if __name__ == "__main__":
    main()
