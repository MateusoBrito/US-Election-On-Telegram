import pandas as pd
import numpy as np
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
import os
import datetime
from tqdm import tqdm

RESULTS_DIR = "results_2"
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
df = pd.read_csv("../data/youtube_telegram_cross.csv")
log(f"{len(df)} linhas carregadas.")

log("Carregando dataframe de treino...")
df_train = pd.read_csv("../data/sample_yt_te_cross_with_topics.csv")
log(f"{len(df_train)} linhas carregadas.")

# Corrige alinhamento de índices
df_embeddings = df[['video_id']].copy()
df_embeddings['embedding_index'] = np.arange(len(df))
df_train_merged = df_train.merge(df_embeddings, on='video_id', how='left')

missing = df_train_merged['embedding_index'].isna().sum()
log(f"Linhas sem embedding correspondente: {missing}")

train_idx = df_train_merged['embedding_index'].dropna().astype(int).values
y = df_train_merged["interest"].values

# ========== CARREGA EMBEDDINGS ==========
embeddings = np.load("embeddings.npy")
embeddings_reduced = np.load("embeddings_reduced.npy")

X = embeddings[train_idx]
X_reduced = embeddings_reduced[train_idx]

# ========== MODELOS ==========
"""
("Decision Tree (original)", tree.DecisionTreeClassifier(random_state=42),
    {'criterion': ['gini', 'entropy'], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'splitter': ['best', 'random'], 'max_features': [None, 'sqrt', 'log2']}),

("Decision Tree (reduced)", tree.DecisionTreeClassifier(random_state=42),
    {'criterion': ['gini', 'entropy'], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'splitter': ['best', 'random'], 'max_features': [None, 'sqrt', 'log2']}),

("Logistic Regression (original)", LogisticRegression(max_iter=10000, random_state=42),
    {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear', 'lbfgs'], 'class_weight': [None, 'balanced']}),

("Logistic Regression (reduced)", LogisticRegression(max_iter=10000, random_state=42),
    {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear', 'lbfgs'], 'class_weight': [None, 'balanced']}),

("KNN (original)", KNeighborsClassifier(),
    {'n_neighbors': [5, 10, 15, 25], 'metric': ['euclidean', 'manhattan'], 'weights': ['uniform', 'distance']}),

("KNN (reduced)", KNeighborsClassifier(),
    {'n_neighbors': [5, 10, 15, 25], 'metric': ['euclidean', 'manhattan'], 'weights': ['uniform', 'distance']}),
"""
models = [
    ("RF (original)", RandomForestClassifier(random_state=42), 
     {'n_estimators': [100, 200], 'max_depth': [None, 10], 'min_samples_split': [2, 5], 'class_weight': [None, 'balanced']}),    

    ("RF (reduced)", RandomForestClassifier(random_state=42),
     {'n_estimators': [100, 200], 'max_depth': [None, 10], 'min_samples_split': [2, 5],  'class_weight': [None, 'balanced']}),
     
    ("SVM (original)", SVC(random_state=42),
     {'kernel': ['linear'], 'C': [0.1,10], 'gamma': ['scale', 'auto'], 'class_weight': [None, 'balanced']}),

    ("SVM (reduced)", SVC(random_state=42),
     {'kernel': ['linear'], 'C': [0.1,10], 'gamma': ['scale', 'auto'], 'class_weight': [None, 'balanced']}),
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
