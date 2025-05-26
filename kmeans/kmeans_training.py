import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

df = pd.read_csv("TCC_responses.csv") 
columns_to_remove = ['Timestamp', 'Nome', 'Email', 'Em qual período você está ?']
df_clean = df.drop(columns=[col for col in columns_to_remove if col in df.columns])

# Mapeia as respostas
def map_responses(df):
    approved_map = {
        'Sim': 1,
        'Não': 0
    }

    difficulty_map = {
        'Muito Fácil': 1,
        'Fácil': 2,
        'Neutro': 3,
        'Difícil': 4,
        'Muito Difícil': 5
    }

    interest_map = {
        'Muito Desinteressante': 1,
        'Desinteressante': 2,
        'Neutro': 3,
        'Interessante': 4,
        'Muito Interessante': 5
    }

    importance_map = {
        'Sem Importância': 1,
        'Pouco Importante': 2,
        'Neutro': 3,
        'Importante': 4,
        'Muito Importante': 5
    }

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
            nova_coluna = df[col].map(approved_map)
            nova_coluna.name = f"Você foi aprovado - {discipline_code}"
            processed_columns.append(nova_coluna)

        elif 'grau de dificuldade' in col:
            nova_coluna = df[col].map(difficulty_map)
            nova_coluna.name = f"Dificuldade - {discipline_code}"
            processed_columns.append(nova_coluna)

        elif 'considera o assunto' in col:
            nova_coluna = df[col].map(interest_map)
            nova_coluna.name = f"Interesse - {discipline_code}"
            processed_columns.append(nova_coluna)

        elif 'importância' in col:
            nova_coluna = df[col].map(importance_map)
            nova_coluna.name = f"Importância - {discipline_code}"
            processed_columns.append(nova_coluna)

    return pd.concat(processed_columns, axis=1)

df_numeric = map_responses(df_clean)

# Se não for respondido é preenchido com 0
df_numeric = df_numeric.fillna(0)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

# Treina o modelo k-means (k=2)
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(df_scaled)

# Salva
os.makedirs("models", exist_ok=True)
joblib.dump(kmeans, "models/kmeans_model.pkl")
joblib.dump(scaler, "models/kmeans_scaler.pkl")

# Salva
os.makedirs("data", exist_ok=True)
with open("data/columns.txt", "w", encoding="utf-8") as f:
    for col in df_numeric.columns:
        f.write(f"{col}\n")

print("Modelo K-means treinado!")