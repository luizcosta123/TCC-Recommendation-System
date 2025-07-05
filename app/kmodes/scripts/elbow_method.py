from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("../data/form_responses.csv")
data_matrix = data.to_numpy()

k_range = range(1, 11)
costs = []

for k in k_range:
    print(f"Testando k = {k}")
    if k == 1:
        n_samples, n_features = data_matrix.shape
        modes = []
        for j in range(n_features):
            unique, counts = np.unique(data_matrix[:, j], return_counts=True)
            mode = unique[np.argmax(counts)]
            modes.append(mode)
        modes = np.array(modes)
        cost = int(np.sum(data_matrix != modes))
    else:
        km = KModes(
            n_clusters=k,
            init="Huang",
            n_init=5,
            verbose=0,
            random_state=42,
        )
        km.fit_predict(data_matrix)
        cost = km.cost_
    costs.append(cost)

ks = np.array(list(k_range))
vals = np.array(costs)

line = np.array([ks[-1] - ks[0], vals[-1] - vals[0]])
norm = np.hypot(*line)
distances = np.abs(line[1] * (ks - ks[0]) - line[0] * (vals - vals[0])) / norm
optimal_k = ks[np.argmax(distances)]

print(f"k ideal sugerido = {optimal_k}")

plt.figure(figsize=(8, 6))

plt.plot(k_range, costs, "bo-", linewidth=2, markersize=8)
plt.axvline(optimal_k, color="red", linestyle="--", label=f"k ideal = {optimal_k}")
plt.legend()

plt.xlabel("Número de Clusters (k)", fontsize=12)
plt.ylabel("Custo (soma de dissimilaridades intra-cluster)", fontsize=12)
plt.title("Método do Cotovelo aplicado ao K-Modes", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.xticks(k_range)

for k, cost in zip(k_range, costs):
    plt.annotate(f"{cost:.0f}", (k, cost), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=9)

plt.tight_layout()
plt.show()