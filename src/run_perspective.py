import pandas as pd
import os
from tqdm import tqdm
import time
import numpy as np
from googleapiclient import discovery
from googleapiclient.errors import HttpError

# --- 1. Configurações ---
API_KEY = 'AIzaSyBZIpJnnUEdokOyMV8Wg3iPWyWgfJjHKOY' 
INPUT_FILE = 'sample_yt_te_cross_with_topics.csv' 
OUTPUT_FILE = 'data/videos_with_perspective_scores_60k.csv' 
TEXT_COLUMN = 'text' 
NUM_ROWS_TO_PROCESS = None 
# ----------------------------------------------------

print("Construindo o cliente da Perspective API...")
try:
    client = discovery.build(
      "commentanalyzer",
      "v1alpha1",
      developerKey=API_KEY,
      discoveryServiceUrl="https"
                          "://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
      static_discovery=False,
    )
except Exception as e:
    print("ERRO: Não foi possível construir o cliente da API.")
    print(f"Erro: {e}")
    exit()

def get_perspective_scores(text_to_analyze):
    time.sleep(1.1) 
    
    analyze_request = {
        'comment': {'text': text_to_analyze},
        'requestedAttributes': {
            'TOXICITY': {},
            'SEVERE_TOXICITY': {},
            'IDENTITY_ATTACK': {},
            'INSULT': {},
            'THREAT': {},
            'OBSCENE': {}
        },
        'languages': ['en'] 
    }

    try:
        response = client.comments().analyze(body=analyze_request).execute()
        
        scores = {}
        for attr, value in response['attributeScores'].items():
            scores[attr.lower()] = value['summaryScore']['value']
        return scores
    
    except HttpError as e:
        print(f"\nErro na chamada da API (ex: quota, texto inválido): {e}\n")
        return {
            'toxicity': np.nan, 'severe_toxicity': np.nan, 'identity_attack': np.nan, 
            'insult': np.nan, 'threat': np.nan, 'obscene': np.nan
        }
    except Exception as e:
        print(f"\nErro inesperado (ex: rede): {e}\n")
        return {
            'toxicity': np.nan, 'severe_toxicity': np.nan, 'identity_attack': np.nan, 
            'insult': np.nan, 'threat': np.nan, 'obscene': np.nan
        }


print(f"Carregando dados de: {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE, nrows=NUM_ROWS_TO_PROCESS)

df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna('') 
total_rows = len(df) 

print(f"Iniciando processamento de {total_rows} textos da coluna '{TEXT_COLUMN}'...")

estimated_hours = (total_rows * 1.1 / 3600)
rounded_hours = round(estimated_hours, 2)
print(f"Tempo total estimado: {rounded_hours} horas.")


all_results = []
for text in tqdm(df[TEXT_COLUMN], total=total_rows, desc="Analisando textos"):
    scores = get_perspective_scores(text)
    all_results.append(scores)

print("Processamento concluído. Combinando resultados...")

df_scores = pd.DataFrame(all_results, index=df.index)

df_scores = df_scores.rename(columns={
    'toxicity': 'perspective_toxicity',
    'severe_toxicity': 'perspective_severe_toxicity',
    'identity_attack': 'perspective_identity_attack',
    'insult': 'perspective_insult',
    'threat': 'perspective_threat',
    'obscene': 'perspective_obscene'
})

df_final = pd.concat([df, df_scores], axis=1)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
print(f"Salvando resultados em {OUTPUT_FILE}...")
df_final.to_csv(OUTPUT_FILE, index=False)

print("Análise com Perspective API concluída com sucesso!")