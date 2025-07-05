import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pickle
import json

with open('../training/models/modelo_kmodes.pkl', 'rb') as f:
    km = pickle.load(f)

clusters = np.load('../training/models/clusters.npy')

with open('../training/models/metadata_modelo.json', 'r') as f:
    metadata = json.load(f)

data = pd.read_csv('../data/form_responses.csv')
data['cluster'] = clusters

k = metadata['n_clusters']

cluster_colors = {
    0: "#00ffd5",
    1: "#ff8800",
    2: "#2200ff"
}

importancia_labels = ['Muito Importante', 'Importante', 'Neutro', 'Pouco Importante', 'Sem Importância']
fig, ax5 = plt.subplots(figsize=(8, 6))
matriz_importancia = []

for cluster_id in range(k):
    cluster_data = data[data['cluster'] == cluster_id]
    contagens = []

    for nivel in importancia_labels:
        total = 0
        for col in data.columns:
            if 'importância' in col.lower() and col != 'cluster':
                total += (cluster_data[col] == nivel).sum()
        contagens.append(total)

    total_respostas = sum(contagens)
    if total_respostas > 0:
        contagens = [c/total_respostas*100 for c in contagens]
    matriz_importancia.append(contagens)

sns.heatmap(matriz_importancia,
            xticklabels=importancia_labels,
            yticklabels=[f'Cluster {i}' for i in range(k)],
            cmap='PuRd',
            annot=True,
            fmt='.1f',
            cbar_kws={'label': 'Percentual (%)'},
            ax=ax5)

ax5.set_title('Importância Percebida', fontsize=12, fontweight='bold')
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
fig.tight_layout()
plt.show()

interesse_labels = ['Muito Interessante', 'Interessante', 'Neutro', 'Desinteressante', 'Muito Desinteressante']
fig, ax4 = plt.subplots(figsize=(8, 6))
matriz_interesse = []

for cluster_id in range(k):
    cluster_data = data[data['cluster'] == cluster_id]
    contagens = []

    for nivel in interesse_labels:
        total = 0
        for col in data.columns:
            if 'interess' in col.lower() and col != 'cluster':
                total += (cluster_data[col] == nivel).sum()
        contagens.append(total)

    total_respostas = sum(contagens)
    if total_respostas > 0:
        contagens = [c/total_respostas*100 for c in contagens]
    matriz_interesse.append(contagens)

sns.heatmap(matriz_interesse,
            xticklabels=interesse_labels,
            yticklabels=[f'Cluster {i}' for i in range(k)],
            cmap='YlGnBu',
            annot=True,
            fmt='.1f',
            cbar_kws={'label': 'Percentual (%)'},
            ax=ax4)

ax4.set_title('Interesse Percebido', fontsize=12, fontweight='bold')
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
fig.tight_layout()
plt.show()

dificuldade_labels = ['Muito Fácil', 'Fácil', 'Neutro', 'Difícil', 'Muito Difícil']
fig, ax3 = plt.subplots(figsize=(8, 6))
matriz_dificuldade = []

for cluster_id in range(k):
    cluster_data = data[data['cluster'] == cluster_id]
    contagens = []

    for nivel in dificuldade_labels:
        total = 0
        for col in data.columns:
            if 'dificuldade' in col.lower() and col != 'cluster':
                total += (cluster_data[col] == nivel).sum()
        contagens.append(total)

    total_respostas = sum(contagens)
    if total_respostas > 0:
        contagens = [c/total_respostas*100 for c in contagens]
    matriz_dificuldade.append(contagens)

sns.heatmap(matriz_dificuldade,
            xticklabels=dificuldade_labels,
            yticklabels=[f'Cluster {i}' for i in range(k)],
            cmap='coolwarm',
            annot=True,
            fmt='.1f',
            cbar_kws={'label': 'Percentual (%)'},
            ax=ax3)

ax3.set_title('Dificuldade Percebida', fontsize=12, fontweight='bold')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
fig.tight_layout()
plt.show()

def analyze_centroid(centroid_values):
    stats = {
        'aprovacao': {'Sim': 0, 'Não': 0, 'Não cursou': 0},
        'dificuldade': {'Muito Fácil': 0, 'Fácil': 0, 'Neutro': 0, 'Difícil': 0, 'Muito Difícil': 0},
        'interesse': {'Muito Interessante': 0, 'Interessante': 0, 'Neutro': 0, 'Desinteressante': 0, 'Muito Desinteressante': 0},
        'importancia': {'Muito Importante': 0, 'Importante': 0, 'Neutro': 0, 'Pouco Importante': 0, 'Sem Importância': 0}
    }

    for i, col in enumerate(data.columns[:-1]):
        if i < len(centroid_values):
            value = centroid_values[i]

            if 'aprovado' in col:
                if value in stats['aprovacao']:
                    stats['aprovacao'][value] += 1
            elif 'dificuldade' in col.lower():
                if value in stats['dificuldade']:
                    stats['dificuldade'][value] += 1
            elif 'interess' in col.lower():
                if value in stats['interesse']:
                    stats['interesse'][value] += 1
            elif 'importância' in col.lower():
                if value in stats['importancia']:
                    stats['importancia'][value] += 1

    return stats

courses = []
for col in data.columns[:-1]:
    if 'aprovado' in col:
        disciplina = col.split('\n')[0]
        if disciplina not in courses:
            courses.append(disciplina)

cluster_profiles = {}

for i in range(k):
    centroid = km.cluster_centroids_[i]
    stats = analyze_centroid(centroid)
    
    total_courses = sum(stats['aprovacao'].values())
    if total_courses > 0:
        aprovacao_pct = (stats['aprovacao']['Sim'] / total_courses) * 100
        nao_cursou_pct = (stats['aprovacao']['Não cursou'] / total_courses) * 100

    dif_predominante = max(stats['dificuldade'].items(), key=lambda x: x[1])[0]
    int_predominante = max(stats['interesse'].items(), key=lambda x: x[1])[0]
    imp_predominante = max(stats['importancia'].items(), key=lambda x: x[1])[0]

    cluster_profiles[f'Cluster {i}'] = {
        'stats': stats,
        'aprovacao_pct': aprovacao_pct,
        'nao_cursou_pct': nao_cursou_pct,
        'dif_predominante': dif_predominante,
        'int_predominante': int_predominante,
        'imp_predominante': imp_predominante
    }

for i in range(k):
    cluster_data = data[data['cluster'] == i]

    approvals = 0
    total_courses = 0
    for col in data.columns:
        if 'aprovado' in col and col != 'cluster':
            approvals += (cluster_data[col] == 'Sim').sum()
            total_courses += (cluster_data[col] != 'Não cursou').sum()

    taxa_aprovacao = (approvals / total_courses * 100) if total_courses > 0 else 0

    piores_courses = []
    for disciplina in courses:
        cols_aprovacao = [col for col in data.columns if disciplina in col and 'aprovado' in col]
        if cols_aprovacao:
            aprovados = (cluster_data[cols_aprovacao[0]] == 'Sim').sum()
            total = len(cluster_data[cluster_data[cols_aprovacao[0]] != 'Não cursou'])
            if total > 0:
                taxa = aprovados / total * 100
                piores_courses.append((disciplina.split(' - ')[0], taxa))

    piores_courses.sort(key=lambda x: x[1])

data_encoded = data.copy()
le = LabelEncoder()
for col in data_encoded.columns[:-1]:
    data_encoded[col] = le.fit_transform(data_encoded[col].astype(str))

fig, ax = plt.subplots(figsize=(20, 10))

matriz_aprovacao = []
for cluster_id in range(k):
    cluster_data = data[data['cluster'] == cluster_id]
    taxas = []

    for disciplina in courses[:70]:
        cols_aprovacao = [col for col in data.columns if disciplina in col and 'aprovado' in col]

        if cols_aprovacao:
            aprovados = (cluster_data[cols_aprovacao[0]] == 'Sim').sum()
            total = len(cluster_data[cluster_data[cols_aprovacao[0]] != 'Não cursou'])
            taxa = (aprovados / total * 100) if total > 0 else 0
            taxas.append(taxa)

    matriz_aprovacao.append(taxas)

sns.heatmap(
    matriz_aprovacao,
    xticklabels=[d.split(' - ')[0][:12] for d in courses[:70]],
    yticklabels=[f'Cluster {i}' for i in range(k)],
    cmap='RdYlGn',
    annot=True,
    fmt='.0f',
    cbar_kws={'label': 'Taxa de Aprovação (%)'},
    vmin=0,
    vmax=100
)

plt.title('Taxa de Aprovação por Cluster e Disciplina (%)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()