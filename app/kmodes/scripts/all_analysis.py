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

cluster_colors = {
    0: "#00ffd5",
    1: "#ff8800",
    2: "#2200ff"
}

k = metadata['n_clusters']

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

fig, ax1 = plt.subplots(figsize=(6, 6))
cluster_counts = [len(data[data['cluster'] == i]) for i in range(k)]
ax1.pie(cluster_counts, labels=[f'Cluster {i}\n({c} alunos)' for i, c in enumerate(cluster_counts)],
        colors=list(cluster_colors.values()), autopct='%1.1f%%', startangle=90)
ax1.set_title('Distribuição dos Alunos por Cluster', fontsize=12, fontweight='bold')
fig.tight_layout()
plt.show()

fig, ax2 = plt.subplots(figsize=(6, 6))
approval_rates = []
for i in range(k):
    cluster_data = data[data['cluster'] == i]
    approvals = 0
    total = 0
    for col in data.columns:
        if 'aprovado' in col and col != 'cluster':
            approvals += (cluster_data[col] == 'Sim').sum()
            total += (cluster_data[col] != 'Não cursou').sum()
    taxa = (approvals / total * 100) if total > 0 else 0
    approval_rates.append(taxa)

bars = ax2.bar(range(k), approval_rates, color=list(cluster_colors.values()))
ax2.set_xlabel('Clusters')
ax2.set_ylabel('Taxa de Aprovação (%)')
ax2.set_title('Taxa Geral de Aprovação por Cluster', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 100)

for i, (bar, taxa) in enumerate(zip(bars, approval_rates)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 4,
             f'{taxa:.1f}%', ha='center', va='bottom')
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

fig, ax6 = plt.subplots(figsize=(8, 6), subplot_kw={'projection': 'polar'})

categorias = ['Taxa de Aprovação', 'Facilidade', 'Interesse', 'Importância', 'Cobertura']
num_vars = len(categorias)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

for i in range(k):
    valores = []

    valores.append(approval_rates[i])

    profile = cluster_profiles[f'Cluster {i}']

    facilidade = 0
    if profile['dif_predominante'] == 'Muito Fácil':
        facilidade = 90
    elif profile['dif_predominante'] == 'Fácil':
        facilidade = 70
    elif profile['dif_predominante'] == 'Neutro':
        facilidade = 50
    elif profile['dif_predominante'] == 'Difícil':
        facilidade = 30
    elif profile['dif_predominante'] == 'Muito Difícil':
        facilidade = 10
    else:
        facilidade = 0
    valores.append(facilidade)

    interesse = 0
    if profile['int_predominante'] == 'Muito Interessante':
        interesse = 90
    elif profile['int_predominante'] == 'Interessante':
        interesse = 70
    elif profile['int_predominante'] == 'Neutro':
        interesse = 50
    elif profile['int_predominante'] == 'Desinteressante':
        interesse = 30
    elif profile['int_predominante'] == 'Muito Desinteressante':
        interesse = 10
    else:
        interesse = 0
    valores.append(interesse)

    importancia = 0
    if profile['imp_predominante'] == 'Muito Importante':
        importancia = 90
    elif profile['imp_predominante'] == 'Importante':
        importancia = 70
    elif profile['imp_predominante'] == 'Neutro':
        importancia = 50
    elif profile['imp_predominante'] == 'Pouco Importante':
        importancia = 30
    elif profile['imp_predominante'] == 'Sem Importância':
        importancia = 10
    else:
        importancia = 0
    valores.append(importancia)

    cobertura = 100 - profile['nao_cursou_pct']
    valores.append(cobertura)

    valores_plot = valores + [valores[0]]
    ax6.plot(angles, valores_plot, 'o-', linewidth=2,
             label=f'Cluster {i}', color=cluster_colors[i])
    ax6.fill(angles, valores_plot, alpha=0.25, color=cluster_colors[i])

ax6.set_theta_offset(np.pi / 2)
ax6.set_theta_direction(-1)
ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(categorias)
ax6.tick_params(axis='x', pad=15)
ax6.set_ylim(0, 100)
ax6.set_title('Perfil Comparativo dos Clusters', fontsize=12, fontweight='bold', pad=20)
ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax6.grid(True)
fig.tight_layout()
plt.show()
