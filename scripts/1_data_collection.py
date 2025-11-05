
import requests
import pandas as pd
import os
from datetime import datetime

def fetch_fear_greed():
    print("Récupération des données Fear & Greed depuis alternative.me...")

    url = "https://api.alternative.me/fng/?limit=0"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Erreur API : {e}")
        return None

    data = response.json()['data']
    if not data:
        print("Aucune donnée reçue.")
        return None

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['value'] = df['value'].astype(int)
    df = df.rename(columns={'value_classification': 'classification'})
    df = df[['timestamp', 'value', 'classification']]
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Sauvegarde
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'fear_greed_data.csv')
    df.to_csv(csv_path, index=False)

    print(f"Succès ! {len(df)} jours collectés.")
    print(f"Fichier sauvgardé : data/fear_greed_data.csv")
    print("\nDernières 5 lignes :")
    print(df.tail(5)[['timestamp', 'value', 'classification']].to_string(index=False))

    return df

if __name__ == "__main__":
    fetch_fear_greed()
