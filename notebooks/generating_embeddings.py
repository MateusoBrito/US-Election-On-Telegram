import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# Carregue seus dados pré-processados
df = pd.read_csv("../data/preprocessed_english_titles.csv")
use_df = df[df['clean_text'].notna()].copy()
docs = use_df['clean_text'].tolist()

# Escolha o modelo de embedding
print("Carregando o modelo de embedding...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Gere os embeddings (esta é a parte demorada)
print("Iniciando a geração dos embeddings. Isso pode demorar...")
embeddings = embedding_model.encode(docs, show_progress_bar=True)

# Salve os embeddings em um arquivo
print("Salvando embeddings no arquivo 'embeddings.npy'...")
np.save("embeddings.npy", embeddings)

print("Embeddings salvos com sucesso!")