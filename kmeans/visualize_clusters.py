import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv("TCC_responses.csv")
columns_to_remove = ['Timestamp', 'Nome', 'Email', 'Em qual período você está ?']
df_clean = df.drop(columns=[col for col in columns_to_remove if col in df.columns])

def map_responses(df):
    approved_map = {'Sim': 1, 'Não': 0}
    difficulty_map = {'Muito Fácil': 1, 'Fácil': 2, 'Neutro': 3, 'Difícil': 4, 'Muito Difícil': 5}
    interest_map = {'Muito Desinteressante': 1, 'Desinteressante': 2, 'Neutro': 3, 'Interessante': 4, 'Muito Interessante': 5}
    importance_map = {'Sem Importância': 1, 'Pouco Importante': 2, 'Neutro': 3, 'Importante': 4, 'Muito Importante': 5}

    processed_columns = []

    for col in df.columns:
        if "CPA" not in col:
            continue

        discipline_code = None
        for part in col.split():
            if part.startswith("CPA"):
                discipline_code = part
                break

        if not discipline_code:
            continue

        if 'aprovado na disciplina' in col:
            new_column = df[col].map(approved_map)
            new_column.name = f"Você foi aprovado - {discipline_code}"
            processed_columns.append(new_column)

        elif 'grau de dificuldade' in col:
            new_column = df[col].map(difficulty_map)
            new_column.name = f"Dificuldade - {discipline_code}"
            processed_columns.append(new_column)

        elif 'considera o assunto' in col:
            new_column = df[col].map(interest_map)
            new_column.name = f"Interesse - {discipline_code}"
            processed_columns.append(new_column)

        elif 'importância' in col:
            new_column = df[col].map(importance_map)
            new_column.name = f"Importância - {discipline_code}"
            processed_columns.append(new_column)

    return pd.concat(processed_columns, axis=1)

df_numeric = map_responses(df_clean).fillna(0)

scaler = joblib.load("models/kmeans_scaler.pkl")
model = joblib.load("models/kmeans_model.pkl")
df_scaled = scaler.transform(df_numeric)
clusters = model.predict(df_scaled)

# Reduz para 2 dimensões
pca = PCA(n_components=2)
reduced = pca.fit_transform(df_scaled)

# Distribuição de alunos por cluster
plt.figure(figsize=(7, 5))
unique_clusters, counts = np.unique(clusters, return_counts=True)
plt.bar(unique_clusters, counts, color=['red', 'blue', 'green', 'orange'][:len(unique_clusters)])
plt.title("Distribuição de Alunos por Cluster")
plt.xlabel("Cluster")
plt.ylabel("Número de Alunos")
for i, count in enumerate(counts):
    plt.text(unique_clusters[i], count + 0.1, str(count), ha='center')

# Visualização dos clusters
plt.figure(figsize=(7, 5))
plt.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap='rainbow', alpha=0.7)
for i, cluster in enumerate(clusters):
    plt.annotate(str(cluster), (reduced[i, 0], reduced[i, 1]),
                 xytext=(3, 3), textcoords='offset points', fontsize=8, alpha=0.7)
plt.title("Visualização dos Clusters de Alunos")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.grid(True)
plt.show()