from kmodes.kmodes import KModes
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime

data = pd.read_csv('../data/form_responses.csv')
data_matrix = data.to_numpy()

k = 3

km = KModes(n_clusters=k, init='Huang', n_init=10000, verbose=1)
clusters = km.fit_predict(data_matrix)

with open('models/modelo_kmodes.pkl', 'wb') as f:
    pickle.dump(km, f)
np.save('models/clusters.npy', clusters)

metadata = {
    'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'n_clusters': k,
    'n_amostras': len(data),
    'n_features': data.shape[1],
    'cost': float(km.cost_),
    'n_init': 10000,
    'init_method': 'Huang',
    'cluster_sizes': np.bincount(clusters).tolist()
}

with open('models/metadata_modelo.json', 'w') as f:
    json.dump(metadata, f, indent=4)
