import pandas as pd
from detoxify import Detoxify
import os
from tqdm import tqdm

# --- Configuração ---
INPUT_FILE = 'processed_df.csv'
OUTPUT_FILE = 'data/videos_with_toxics.csv' 
TEXT_COLUMN = 'clean_text' 
BATCH_SIZE = 512
# --------------------

print("Carregando o modelo Detoxify ('original')...")

model = Detoxify('original', device='cuda') 

print(f"Carregando dados de {INPUT_FILE} ")
# Adiciona 'nrows' para ler apenas as primeiras 50k linhas
df = pd.read_csv(INPUT_FILE)
# --------------------

# Garante que a coluna de texto não tenha valores nulos (NaN)
df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna('') 
texts_list = df[TEXT_COLUMN].tolist()
total_texts = len(texts_list) # Agora, total_texts será 50000

print(f"Iniciando processamento de {total_texts} textos em lotes de {BATCH_SIZE}...")

# Processa em lotes com tqdm
all_results = []
for i in tqdm(range(0, total_texts, BATCH_SIZE), desc="Processando Lotes"):
    batch = texts_list[i : i + BATCH_SIZE]
    
    predictions = model.predict(batch)
    
    all_results.append(pd.DataFrame(predictions))

print("Processamento dos lotes concluído. Combinando resultados...")

df_scores = pd.concat(all_results).reset_index(drop=True)

if len(df_scores) == len(df):
    df_final = pd.concat([df, df_scores], axis=1)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print(f"Salvando resultados em {OUTPUT_FILE}...")
    df_final.to_csv(OUTPUT_FILE, index=False)
    print("Etapa 1 (Detoxify) concluída com sucesso!")
else:
    print(f"Erro: O número de resultados ({len(df_scores)}) não bate com o de entradas ({len(df)}).")