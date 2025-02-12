import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, precision_score
import itertools

# Funzione per aggiungere rumore ai dati
def add_noise(X, noise_level=0.1):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

# Funzione per selezionare le migliori K feature
def select_best_features(X, y, k=20):
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new

# Funzione per eseguire Gradient Boosting con K-Fold CV utilizzando Log Loss e altre metriche
def evaluate_gb(X, y, k_folds=5, repeats=3):
    scores_all = {"log_loss": [], "accuracy": [], "precision": [], "recall": []}
    for seed in range(repeats):
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        scores_all["log_loss"].extend(-cross_val_score(gb_model, X, y, cv=kf, scoring='neg_log_loss'))
        scores_all["accuracy"].extend(cross_val_score(gb_model, X, y, cv=kf, scoring='accuracy'))
        scores_all["precision"].extend(cross_val_score(gb_model, X, y, cv=kf, scoring=make_scorer(precision_score, average="macro", zero_division=1)))
        scores_all["recall"].extend(cross_val_score(gb_model, X, y, cv=kf, scoring='recall_macro'))

    print(f"Cross-Validation ({k_folds}-fold, {repeats} ripetizioni):")
    for metric, values in scores_all.items():
        print(f"{metric.capitalize()}: Media={np.mean(values):.4f}, Dev.Std={np.std(values):.4f}")
    return scores_all

# Funzione per eseguire Gradient Boosting con K-Fold CV e ricerca degli iperparametri
def cv_GB(X, y, k_folds=5, repeats=3):
    # Definiamo il grid di iperparametri
    param_grid = {
        'n_estimators': [50, 100],         # Rimuoviamo 200
        'learning_rate': [0.01, 0.1],        # Rimuoviamo 0.2
        'max_depth': [3, 5]                # Rimuoviamo 7
    }
    gb_model = GradientBoostingClassifier(random_state=42)
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
            gb_model,
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
                # Per log_loss, invertiamo il segno
                if metric == "log_loss":
                    scores_all[metric].append(-results[metric_name].values)
                else:
                    scores_all[metric].append(results[metric_name].values)
    print("Best parameters (GB):", grid_search.best_params_)
    for metric, values in scores_all.items():
        values = np.array(values)
        mean_values = np.mean(values, axis=0)
        print(f"{metric.capitalize()} Media: {np.mean(mean_values):.4f}")

    # Generazione grafico per GridSearchCV: creiamo etichette esplicite per ogni combinazione
    combinations = list(itertools.product(param_grid['n_estimators'], param_grid['learning_rate'], param_grid['max_depth']))
    labels_hyper = [f"n_est={n}, lr={lr}, md={md}" for (n, lr, md) in combinations]

    plt.figure(figsize=(10,6))
    for metric in scoring.keys():
        values = np.array(scores_all[metric])
        mean_metric = np.mean(values, axis=0)
        plt.plot(range(len(mean_metric)), mean_metric, marker='o', label=metric.capitalize())
    plt.xticks(ticks=range(len(labels_hyper)), labels=labels_hyper, rotation=45)
    plt.xlabel('Combinazioni di iperparametri')
    plt.ylabel('Metriche di valutazione')
    plt.title('Gradient Boosting - Effetto degli iperparametri con K-Fold CV')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
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

# Test sugli iperparametri con GridSearchCV
#cv_GB(X_filtered, y)

# Test con diversi livelli di rumore e feature selection
noise_levels = [0.0, 0.1, 0.2]
k_values = [X_filtered.shape[1], 30, 10]
results = {}
from joblib import Parallel, delayed

def eval_for_params(noise, k, X_filtered, y):
    # Aggiunge rumore ai dati
    X_noisy = add_noise(X_filtered, noise_level=noise)
    # Seleziona le migliori k feature
    X_selected = select_best_features(X_noisy, y, k=k)
    # Valuta il modello di Gradient Boosting con K-Fold CV
    scores = evaluate_gb(X_selected, y)
    return (noise, k, scores)

# Lista di combinazioni (noise, k) da testare
combinations = [(noise, k) for noise in noise_levels for k in k_values]

# Esegue in parallelo la funzione per ogni combinazione
results_list = Parallel(n_jobs=-1)(delayed(eval_for_params)(noise, k, X_filtered, y)
                                   for noise, k in combinations)

# Ricostruisce il dizionario dei risultati
results = { (noise, k): scores for noise, k, scores in results_list }

# Creazione del grafico di confronto per i test con rumore e feature selection
plt.figure(figsize=(12, 6))
# Generiamo etichette esplicite per ogni combinazione di rumore e feature selection
labels = [f"Noise {noise}, K {k}" for noise in noise_levels for k in k_values]
data = [results[(noise, k)]["log_loss"] for noise in noise_levels for k in k_values]
sns.boxplot(data=data)
plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
plt.title("Effetto di Rumore e Feature Selection sulla Log Loss di Gradient Boosting")
plt.ylabel("Log Loss")
plt.xlabel("Livello di Rumore e Numero di Feature Selezionate")
plt.grid(True)
plt.tight_layout()
plt.show()

# Stampa delle statistiche per i test con rumore e feature selection
print("Statistiche dei risultati:")
for noise in noise_levels:
    for k in k_values:
        print(f"Noise {noise}, K {k}: Media={np.mean(results[(noise, k)]['log_loss']):.4f}, Dev.Std={np.std(results[(noise, k)]['log_loss']):.4f}")
