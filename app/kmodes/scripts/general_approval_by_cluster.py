import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

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