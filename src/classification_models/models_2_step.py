import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import os
import datetime
from tqdm import tqdm

RESULTS_DIR = "results_2_step"
os.makedirs(RESULTS_DIR, exist_ok=True)

def log(msg):
    """Salva log com timestamp no arquivo geral"""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f"{RESULTS_DIR}/log.txt", "a") as f:
        f.write(f"[{ts}] {msg}\n")
    print(msg)

def run_model(name, model, param_grid, X, y, cv_splits=5):
    log(f"\n Rodando modelo: {name}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    strat_kfold = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    clf = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=strat_kfold,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )

    clf.fit(X_scaled, y)
    best_model = clf.best_estimator_
    log(f"Melhores hiperparâmetros encontrados: {clf.best_params_}")

    # Salva resultados detalhados em arquivo
    file_path = f"{RESULTS_DIR}/{name.replace(' ', '_')}.txt"
    with open(file_path, "w") as f:
        f.write(f"=== {name} ===\n")
        f.write(f"Melhores parâmetros: {clf.best_params_}\n\n")

        reports = []
        for fold, (train_idx, test_idx) in enumerate(strat_kfold.split(X_scaled, y), 1):
            y_pred = best_model.fit(X_scaled[train_idx], y[train_idx]).predict(X_scaled[test_idx])
            report = classification_report(y[test_idx], y_pred, digits=3, output_dict=True)
            reports.append(report)

            f.write(f"\n--- Fold {fold} ---\n")
            f.write(classification_report(y[test_idx], y_pred, digits=3))
            f.write("\n")

        avg_f1_macro = np.mean([r['macro avg']['f1-score'] for r in reports])
        std_f1_macro = np.std([r['macro avg']['f1-score'] for r in reports])
        f.write(f"\nMédia F1 Macro: {avg_f1_macro:.3f} ± {std_f1_macro:.3f}\n")

    log(f"✅ Finalizado: {name} | Média F1 Macro: {avg_f1_macro:.3f}")
    return best_model

# ========== CARREGA DADOS ==========
log("Carregando dataframe completo...")
df = pd.read_csv("../data/youtube_telegram_predicted_interest.csv")
size_df = len(df)

df = df[df['interest_pred'] == 1]
log(f"{size_df} linhas carregadas. {len(df)} linhas de interesse.")

log("Carregando dataframe de treino...")
df_train = pd.read_csv("../sample_yt_te_cross_with_topics.csv")
size_df_train = len(df_train)

df_train = df_train[df_train['interest'] == 1]
log(f"{size_df_train} linhas carregadas. {len(df_train)} linhas de interesse.")

# Corrige alinhamento de índices
df_embeddings = df[['video_id']].copy()
df_embeddings['embedding_index'] = np.arange(len(df))
df_train_merged = df_train.merge(df_embeddings, on='video_id', how='left')

missing = df_train_merged['embedding_index'].isna().sum()
log(f"Linhas sem embedding correspondente: {missing}")

df_train_valid = df_train_merged.dropna(subset=['embedding_index']).copy()
df_train_valid['embedding_index'] = df_train_valid['embedding_index'].astype(int)

train_idx = df_train_valid['embedding_index'].values
y = df_train_valid['topic'].values

# ========== CARREGA EMBEDDINGS ==========
embeddings = np.load("embeddings.npy")
embeddings_reduced = np.load("embeddings_reduced.npy")

X = embeddings[train_idx]
X_reduced = embeddings_reduced[train_idx]


# ========== MODELOS ==========
models = [
    ("Logistic Regression (reduced)", LogisticRegression(max_iter=10000, multi_class="auto", random_state=42),
     {'C': [0.1, 1, 10], 
      'class_weight': [None, 'balanced'], 
      'solver': ['lbfgs', 'saga'], 
      'penalty': ['l1', 'l2']}), # l1 requer solver 'saga'

    ("KNN (reduced)", KNeighborsClassifier(),
     {'n_neighbors': [5, 10, 15, 25], 
      'metric': ['euclidean', 'manhattan'], 
      'weights': ['uniform', 'distance'], 
      'p': [1, 2]}), # p=1 (manhattan), p=2 (euclidean)

    ("XGBoost (reduced)", xgb.XGBClassifier(objective="multi:softmax", eval_metric="mlogloss", use_label_encoder=False),
     {'n_estimators': [200, 500], 
      'max_depth': [4, 6, 10], 
      'learning_rate': [0.05, 0.1], 
      'gamma': [0, 0.1], 
      'subsample': [0.8, 1.0]}),

    ("Random Forest (reduced)", RandomForestClassifier(random_state=42),
     {'n_estimators': [200, 500], 
      'max_depth': [10, 20, None], 
      'class_weight': [None, 'balanced'], 
      'min_samples_split': [2, 5, 10]}),

    ("SVM (reduced)", SVC(random_state=42),
     {'kernel': ['linear', 'rbf'], 
      'C': [0.1, 1, 10], 
      'class_weight': [None, 'balanced']}),

    ("MultinomialNB (reduced)", MultinomialNB(),
     {'alpha': [0.01, 0.1, 1.0, 10.0]}),
]

# ========== EXECUÇÃO ==========
summary = []

for name, model, params in tqdm(models, desc="Treinando modelos", ncols=100):
    X_input = X_reduced if "reduced" in name else X
    result = run_model(name, model, params, X_input, y)
    summary.append(result)

# Salva resumo em CSV
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(RESULTS_DIR, "results_summary.csv"), index=False)
log("📊 Resumo salvo em results_summary.csv")
log("🏁 Todos os modelos finalizados.")
