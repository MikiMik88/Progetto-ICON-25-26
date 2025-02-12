import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.preprocessing import StandardScaler

# Funzione per aggiungere rumore ai dati
def add_noise(X, noise_level=0.1):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

# Funzione per selezionare le migliori K feature
def select_best_features(X, y, k=20):
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new

# Funzione per eseguire SVM con K-Fold CV utilizzando Log Loss e altre metriche
from sklearn.metrics import make_scorer, precision_score

# Funzione per eseguire SVM con K-Fold CV utilizzando Log Loss e altre metriche
def evaluate_svm(X, y, k_folds=5, repeats=3):
    scores_all = {"log_loss": [], "accuracy": [], "precision": [], "recall": []}
    for seed in range(repeats):
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        svm_model = SVC(kernel='linear', C=1, probability=True, random_state=42)
        scores_all["log_loss"].extend(-cross_val_score(svm_model, X, y, cv=kf, scoring='neg_log_loss'))
        scores_all["accuracy"].extend(cross_val_score(svm_model, X, y, cv=kf, scoring='accuracy'))
        scores_all["precision"].extend(cross_val_score(svm_model, X, y, cv=kf, scoring=make_scorer(precision_score, average="macro", zero_division=1)))
        scores_all["recall"].extend(cross_val_score(svm_model, X, y, cv=kf, scoring='recall_macro'))

    print(f"Cross-Validation ({k_folds}-fold, {repeats} ripetizioni):")
    for metric, values in scores_all.items():
        print(f"{metric.capitalize()}: Media={np.mean(values):.4f}, Dev.Std={np.std(values):.4f}")
    return scores_all

# Funzione per eseguire SVM con K-Fold CV e ricerca del parametro C
def cv_SVM(X, y, k_folds=5, repeats=3):
    C_values = [0.01, 0.1, 1, 10, 100]
    param_grid = {'C': C_values}
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    scoring = {
        "log_loss": "neg_log_loss",
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro"
    }
    scores_all = {metric: [] for metric in scoring.keys()}
    for seed in range(repeats):
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        grid_search = GridSearchCV(
            svm_model,
            param_grid,
            scoring=scoring,
            cv=kf,
            refit="log_loss",
            return_train_score=True
        )
        grid_search.fit(X, y)
        results = pd.DataFrame(grid_search.cv_results_)
        for metric in scoring.keys():
            metric_name = f"mean_test_{metric}"
            if metric_name in results:
                scores_all[metric].append(-results[metric_name].values if metric == "log_loss" else results[metric_name].values)
    print("Best parameters (SVM):", grid_search.best_params_)
    for metric, values in scores_all.items():
        values = np.array(values)
        mean_values = np.mean(values, axis=0)
        print(f"{metric.capitalize()} Media: {np.mean(mean_values):.4f}")
    plt.figure(figsize=(10,6))
    for metric in scoring.keys():
        values = np.array(scores_all[metric])
        plt.plot(C_values, np.mean(values, axis=0), marker='o', label=metric.capitalize())
    plt.xscale('log')
    plt.xlabel('Valore di C (scala logaritmica)')
    plt.ylabel('Metriche di valutazione')
    plt.title('SVM - Effetto del parametro C con K-Fold CV')
    plt.legend()
    plt.grid(True)
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

# Test sul parametro C
cv_SVM(X_filtered, y)

# Test con diversi livelli di rumore e feature selection
noise_levels = [0.0, 0.05, 0.1, 0.2]
k_values = [X_filtered.shape[1], 30, 20, 10]
results = {}
for noise in noise_levels:
    X_noisy = add_noise(X_filtered, noise_level=noise)
    for k in k_values:
        X_selected = select_best_features(X_noisy, y, k=k)
        scores = evaluate_svm(X_selected, y)
        results[(noise, k)] = scores

# Creazione del grafico di confronto
plt.figure(figsize=(12, 6))
data = [results[(noise, k)]["log_loss"] for noise in noise_levels for k in k_values]
labels = [f"Noise {noise}, K {k}" for noise in noise_levels for k in k_values]
sns.boxplot(data=data)
plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
plt.title("Effetto di Rumore e Feature Selection sulla Log Loss della SVM")
plt.ylabel("Log Loss")
plt.xlabel("Livello di Rumore e Numero di Feature Selezionate")
plt.grid(True)
plt.show()

# Stampa delle statistiche
print("Statistiche dei risultati:")
for noise in noise_levels:
    for k in k_values:
        print(f"Noise {noise}, K {k}: Media={np.mean(results[(noise, k)]['log_loss']):.4f}, Dev.Std={np.std(results[(noise, k)]['log_loss']):.4f}")
