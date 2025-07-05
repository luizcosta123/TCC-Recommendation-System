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

fig, ax1 = plt.subplots(figsize=(6, 6))
cluster_counts = [len(data[data['cluster'] == i]) for i in range(k)]
ax1.pie(cluster_counts, labels=[f'Cluster {i}\n({c} alunos)' for i, c in enumerate(cluster_counts)],
        colors=list(cluster_colors.values()), autopct='%1.1f%%', startangle=90)
ax1.set_title('Distribuição dos Alunos por Cluster', fontsize=12, fontweight='bold')
fig.tight_layout()
plt.show()