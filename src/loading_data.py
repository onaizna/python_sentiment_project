import os
import pandas as pd

def load_multiple_excel(base_path, file_details):
    """
    Carica più file Excel specificati in un dizionario di DataFrame pandas.

    Args:
        base_path (str): Il percorso della directory contenente i file Excel.
        file_details (dict): Un dizionario dove le chiavi sono i nomi desiderati 
                             per i DataFrame (es. 'df_fifa') e i valori sono 
                             i nomi dei file Excel corrispondenti (es. 'file.xlsx').

    Returns:
        dict: Un dizionario dove le chiavi sono quelle specificate in file_details
              e i valori sono i DataFrame pandas caricati. 
              Restituisce un dizionario vuoto se si verificano errori critici.
              Stampa messaggi di errore per file specifici non trovati o illeggibili.
    """
    loaded_data = {}
    print(f"Attempting to load files from base path: {base_path}") # Debug print

    for df_name, filename in file_details.items():
        full_path = os.path.join(base_path, filename)
        try:
            print(f"Loading: {full_path} as '{df_name}'") # Debug print
            loaded_data[df_name] = pd.read_excel(full_path)
            print(f"Successfully loaded: {filename}")
        except FileNotFoundError:
            print(f"Error: File not found at {full_path}. Skipping '{df_name}'.")
        except Exception as e:
            print(f"Error loading {filename} into '{df_name}': {e}. Skipping.")
            
    return loaded_data