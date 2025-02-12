import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Funzione per eseguire AdaBoost con K-Fold CV e ricerca degli iperparametri
def cv_AdaBoost(X, y, k_folds=5, repeats=3):
    param_grid = {
        'n_estimators': [10, 50, 100],
        'learning_rate': [0.01, 0.1, 1.0],
        'estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier(max_depth=5)]
    }

    # Creare una lista di combinazioni di iperparametri
    param_combinations = []
    for n_estimators in param_grid['n_estimators']:
        for learning_rate in param_grid['learning_rate']:
            for estimator in param_grid['estimator']:
                param_combinations.append(f"n={n_estimators}, lr={learning_rate}, d={estimator.max_depth}")

    ada_model = AdaBoostClassifier(random_state=42)
    scoring = {
        "log_loss": "neg_log_loss",
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, average="macro", zero_division=0),
        "recall": make_scorer(recall_score, average="macro")
    }

    scores_all = {metric: [] for metric in scoring.keys()}
    for seed in range(repeats):
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        grid_search = GridSearchCV(
            ada_model,
            param_grid,
            scoring=scoring,
            cv=kf,
            refit="log_loss",
            return_train_score=True,
            n_jobs=-1
        )
        grid_search.fit(X, y)
        results = pd.DataFrame(grid_search.cv_results_)
        for metric in scoring.keys():
            metric_name = f"mean_test_{metric}"
            if metric_name in results:
                scores_all[metric].append(-results[metric_name].values if metric == "log_loss" else results[metric_name].values)

    print("Best parameters (AdaBoost):", grid_search.best_params_)
    for metric, values in scores_all.items():
        values = np.array(values)
        mean_values = np.mean(values, axis=0)
        print(f"{metric.capitalize()} Media: {np.mean(mean_values):.4f}")

    # Plotting dei risultati
    plt.figure(figsize=(12,8))
    for metric in scoring.keys():
        values = np.array(scores_all[metric])
        plt.plot(range(len(values[0])), np.mean(values, axis=0), marker='o', label=metric.capitalize())

    # Sostituire le etichette con i nomi delle combinazioni di iperparametri
    plt.xticks(ticks=range(len(param_combinations)), labels=param_combinations, rotation=90)

    plt.xlabel('Combinazioni di iperparametri')
    plt.ylabel('Metriche di valutazione')
    plt.title('AdaBoost - Effetto degli iperparametri con K-Fold CV')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Per evitare sovrapposizioni nel grafico
    plt.show()

    return grid_search

# Caricamento dataset
df = pd.read_csv("symbipredict_2022.csv")
X = df.drop(columns=["prognosis"]).values
y = df["prognosis"].astype('category').cat.codes.values

# Normalizzazione e rimozione feature costanti
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
selector_var = VarianceThreshold(threshold=0)
X_filtered = selector_var.fit_transform(X_scaled)

# Test sugli iperparametri
cv_AdaBoost(X_filtered, y)
