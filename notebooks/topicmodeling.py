from meu_bertopic import BERTopic
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from bertopic.representation import PartOfSpeech
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import TfidfVectorizer

def run_topic_modeling():
    save_data = 'data_topic_modeling'

    df = pd.read_csv("../data/df_teste.csv") # carrega os títulos já pré-processados
    
    use_df = df[df['clean_text'].notna()] # elimina as tuplas que forem nulas nesta coluna
    use_df = use_df.reset_index(drop=True)

    main_representation = KeyBERTInspired() #extrai as palavras chaves principais

    # Additional ways of representing a topic
    aspect_model1 = PartOfSpeech("en_core_web_sm") #extrai apenas classes gramaticas específicas(como substantivos e adjetivos)
    aspect_model2 = [KeyBERTInspired(top_n_words=10), MaximalMarginalRelevance(diversity=.3)] #gera uma lista das palavras-chaves mais diversificada, evitando termos semelhantes
    
    minhas_stopwords = ['algum', 'soltar', 'acontecer', 'ficar', 'outro','viar','conectar','sugar','jogar','rapaz']

    #vectorizer = TfidfVectorizer(stop_words=minhas_stopwords)
    num = 10 
    params = {
        'nr_topics': num, # número de tópicos
        'language': 'english', 
        'calculate_probabilities': True, # % de um doc em tópicos
        'verbose': False, # explicação detalhada do processo
        'top_n_words': 10, # quantas palavras em cada tópico
        'umap_model': UMAP(n_neighbors=10, 
                  n_components=5, 
                  metric='cosine', 
                  random_state=42),
        'hdbscan_model': HDBSCAN(
            min_cluster_size=50,
            #allow_single_cluster=True,
            min_samples=5,
            #cluster_selection_method='leaf',
            #alpha=1,
            #cluster_selection_method='eom',
            prediction_data=True
        ),
        #'vectorizer_model' : vectorizer,# usa o kmeans como algoritmo de clusterização, fixando o número de tópicos em 10
        'ctfidf_model' : ClassTfidfTransformer(reduce_frequent_words=True)
        }
    
    model = BERTopic(**params) # modelo com os parâmetros definidos

    print(params)
    #print(minhas_stopwords)

    topics, probs = model.fit_transform(use_df['clean_text']) # o modelo é treinado com os textos da coluna clean_text

    #model.save(f"./topicmodeling/{save_data}/kmeans_1", serialization="pickle") # o objeto do modelo treinado é salvo como pickle
    return model, probs, topics