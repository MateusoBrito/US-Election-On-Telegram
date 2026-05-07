import pandas as pd
import os
import time
import numpy as np
from tqdm import tqdm
from googleapiclient import discovery
from googleapiclient.errors import HttpError
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURAÇÕES ---
INPUT_FILE = 'data/youtube_telegram_cross.csv'
FINAL_OUTPUT_FILE = 'data/yt_te_toxicity.csv'
TEXT_COLUMN = 'text'
ID_COLUMN = 'video_id'

API_KEYS = [
    'AIzaSyAMv1H-rM08aua3qi7xOh62eCSNutYuWxo',
    'AIzaSyBZIpJnnUEdokOyMV8Wg3iPWyWgfJjHKOY',
    'AIzaSyANZ0Lc_tsbZ1w9OAw7zVvH-qPDstBPJjs',
    'AIzaSyA0CEAAzYpXmEo6_J3-BUnNdIJECoXIT_g',
    'AIzaSyBXGrSmCx1M87pHPDQrCO7bjIypHyDoNIw',
    'AIzaSyBcwtISygbpwuMc4uBHXK4daUBFvW8UEZ0',
    'AIzaSyDrJTo7uTJvtb2okAUpcwp2xvKZ9-eMFVQ',
    'AIzaSyChuuEMtwi9okB5Ah1JA0WUUzf_6odlQso',
    'AIzaSyCwCnmujMPzRVd9DvemrOAk_qge-EnRWcc',
    'AIzaSyCLnc7Wsp6HNn6Ut0TipqOVrkAJ_nTKAw8',
    'AIzaSyC8YR7GmgRoNrsV5deH5wvrxi_14tAmOBQ'
]


def validate_keys(keys_list):
    print(f"\nVerificando {len(keys_list)} chaves de API antes de começar...")
    valid_keys = []
    
    for key in keys_list:
        try:
            # Tenta construir o cliente
            client = discovery.build(
                "commentanalyzer", "v1alpha1",
                developerKey=key,
                discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                static_discovery=False,
            )
            # Tenta uma requisição leve de teste
            analyze_request = {
                'comment': {'text': 'test'},
                'requestedAttributes': {'TOXICITY': {}},
                'languages': ['en']
            }
            client.comments().analyze(body=analyze_request).execute()
            
            # Se não deu erro, adiciona na lista de válidas
            print(f"Chave ...{key[-4:]}: OK")
            valid_keys.append(key)
            
        except Exception as e:
            print(f"❌ Chave ...{key[-4:]}: FALHOU (Erro: {e})")
    
    return valid_keys

def get_perspective_client(api_key):
    try:
        client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=api_key,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
        )
        return client
    except Exception as e:
        print("ERRO: Não foi possível construir o cliente da API.")
        print(f"Erro: {e}")
        return None # Retorna None em vez de sair abruptamente

def analyze_text(client, text):
    if not client: return None
    try:
        analyze_request = {
        'comment': {'text': text},
        'requestedAttributes': {
            'TOXICITY': {}, 'SEVERE_TOXICITY': {}, 'IDENTITY_ATTACK': {},
            'INSULT': {}, 'THREAT': {}, 'OBSCENE': {}
        },
        'languages': ['en'] 
    }
        # Pausa de segurança para respeitar 1 QPS POR CHAVE
        time.sleep(3.1) 
        response = client.comments().analyze(body=analyze_request).execute()
        
        scores = {}
        for attr, value in response['attributeScores'].items():
            scores['perspective_' + attr.lower()] = value['summaryScore']['value']
        return scores
    except Exception:
        return None

def process_chunk(chunk_df, api_key, chunk_id):
    temp_file = f'data/temp_parte_{chunk_id}.csv'
    client = get_perspective_client(api_key)

    rows_done = 0
    if os.path.exists(temp_file):
        df_done = pd.read_csv(temp_file)
        rows_done = len(df_done)
    else:
        diretorio = os.path.dirname(temp_file)
        if diretorio:
            os.makedirs(diretorio, exist_ok=True)
    
    if rows_done >= len(chunk_df):
        return f"Thread {chunk_id} já estava concluída."
    
    chunk_to_process = chunk_df.iloc[rows_done:]
    print(f"--> Thread {chunk_id} iniciada (Chave: ...{api_key[-4:]}). Processando {len(chunk_to_process)} linhas.")

    batch_results = []

    # Adicionei try/except aqui caso o chunk esteja vazio
    if len(chunk_to_process) > 0:
        for row in tqdm(chunk_to_process.itertuples(index=False), total=len(chunk_to_process), desc=f"T{chunk_id}", position=chunk_id):
            text = getattr(row, TEXT_COLUMN)
            scores = analyze_text(client, text)
            
            row_data = row._asdict()
            if scores:
                row_data.update(scores)
            else:
                cols = ['perspective_toxicity', 'perspective_severe_toxicity', 'perspective_identity_attack', 
                        'perspective_insult', 'perspective_threat', 'perspective_obscene']
                for c in cols:
                    row_data[c] = np.nan
            
            batch_results.append(row_data)

            if len(batch_results) >= 5:
                df_batch = pd.DataFrame(batch_results)
                header = not os.path.exists(temp_file)
                df_batch.to_csv(temp_file, mode='a', header=header, index=False)
                batch_results = []
        
        # Salva o restante final da thread
        if batch_results:
            df_batch = pd.DataFrame(batch_results)
            header = not os.path.exists(temp_file)
            df_batch.to_csv(temp_file, mode='a', header=header, index=False)

    return f"Thread {chunk_id} finalizada."

if __name__ == "__main__":
    valid_api_keys = validate_keys(API_KEYS)
    
    if not valid_api_keys:
        print("Nenhuma chave válida encontrada. Encerrando.")
        exit()
    
    print(f"\n{len(valid_api_keys)} chaves válidas prontas para uso.\n")

    print(f"Carregando df de: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    df[ID_COLUMN] = df[ID_COLUMN].astype(str) 

    df_temp = pd.read_csv('temporario.csv')
    df_temp[ID_COLUMN] = df_temp[ID_COLUMN].astype(str)
    df = df[df[ID_COLUMN].isin(df_temp[ID_COLUMN])]

    print(f"Linhas filtradas (apenas as que estão no temporario): {len(df)}")

   # df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna('')

    # Lógica segura para carregar processados
    if os.path.exists(FINAL_OUTPUT_FILE):
        print(f"Arquivo final encontrado. Filtrando IDs já processados...")
        # Carregamos apenas a coluna ID para economizar memória
        df_results = pd.read_csv(FINAL_OUTPUT_FILE, usecols=[ID_COLUMN])
        df_results[ID_COLUMN] = df_results[ID_COLUMN].astype(str)
        
        initial_len = len(df)
        # Filtra o que NÃO está na lista de processados
        df = df[~df[ID_COLUMN].isin(df_results[ID_COLUMN])]
        print(f"Total original: {initial_len}. Restante para processar: {len(df)}")
    else:
        print("Arquivo final não existe. Processando tudo do zero.")
 
    if len(df) == 0:
        print("Nada a processar.")
        exit()

    print(f"Dividindo o dataframe em {len(valid_api_keys)} partes para processamento paralelo")
    if len(df) == 1:
        chunks = [df]
    else:
        chunks = np.array_split(df, len(valid_api_keys))

    with ThreadPoolExecutor(max_workers=len(valid_api_keys)) as executor:
        futures = []
        for i, api_key in enumerate(valid_api_keys):
            futures.append(executor.submit(process_chunk, chunks[i], api_key, i))
        
        for future in futures:
            print(future.result())
"""
    # JUNÇÃO FINAL (CORRIGIDA)
    print("Unindo arquivos temporários...")
    all_parts = []
    for i in range(len(valid_api_keys)):
        fname = f'data/temp_parte_{i}.csv'
        if os.path.exists(fname):
            all_parts.append(pd.read_csv(fname))
    
    if all_parts:
        new_data_df = pd.concat(all_parts)
        
        # Lógica de salvamento SEGURA (Append se arquivo existe)
        file_exists = os.path.exists(FINAL_OUTPUT_FILE)
        
        print(f"Salvando {len(new_data_df)} novas linhas em {FINAL_OUTPUT_FILE}...")
        
        # Se o arquivo já existe, usamos mode='a' (append) e header=False
        # Se não existe, mode='w' e header=True
        mode = 'a' if file_exists else 'w'
        header = not file_exists
        
        new_data_df.to_csv(FINAL_OUTPUT_FILE, mode=mode, header=header, index=False)
        
        print(f"Sucesso! Dados adicionados.")
    else:
        print("Nenhum arquivo parcial encontrado.")
        """