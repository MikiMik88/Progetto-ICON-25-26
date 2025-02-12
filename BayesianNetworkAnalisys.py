import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, BicScore
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

file_path = "symbipredict_2022.csv"
df = pd.read_csv(file_path)

malattie_mapping = dict(enumerate(df['prognosis'].astype('category').cat.categories))
malattie_mapping_inv = {v: k for k, v in malattie_mapping.items()}

df['prognosis'] = df['prognosis'].astype('category').cat.codes

X_all = df.drop(columns=['prognosis'])
y = df['prognosis']

selector_var = VarianceThreshold(threshold=0)
X_filtered_array = selector_var.fit_transform(X_all)
filtered_features = X_all.columns[selector_var.get_support()]

num_features = 131
selector_kbest = SelectKBest(f_classif, k=num_features)
X_selected_array = selector_kbest.fit_transform(X_filtered_array, y)
selected_features = list(filtered_features[selector_kbest.get_support()])

reduced_df = pd.concat([df['prognosis'], df[selected_features]], axis=1)

train_df, test_df = train_test_split(reduced_df, test_size=0.2, random_state=42, stratify=reduced_df['prognosis'])

def ottimizza_struttura_rete(data):
    hc = HillClimbSearch(data)
    best_model = hc.estimate(scoring_method=BicScore(data))
    return BayesianNetwork(best_model.edges())

model = ottimizza_struttura_rete(train_df)
model.fit(train_df, estimator=MaximumLikelihoodEstimator)

y_true = test_df['prognosis'].values
y_pred = []
for _, row in test_df.iterrows():
    evidence = {s: int(row[s]) for s in selected_features}
    y_pred.append(model.predict(pd.DataFrame([evidence]))['prognosis'][0])

acc = accuracy_score(y_true, y_pred)
class_report = classification_report(y_true, y_pred, target_names=[malattie_mapping[i] for i in range(len(malattie_mapping))])
cm = confusion_matrix(y_true, y_pred)

print("\nAccuratezza sul test set:", acc)
print("\nReport di classificazione:")
print(class_report)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[malattie_mapping[i] for i in range(len(malattie_mapping))],
            yticklabels=[malattie_mapping[i] for i in range(len(malattie_mapping))])
plt.xlabel('Predetto')
plt.ylabel('Reale')
plt.title("Matrice di Confusione")
plt.show()