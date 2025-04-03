import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Données
data = {
    "Marque": ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"],
    "Auto": [7.5, 4.0, 6.9, 8.6, 6.3, 9.3, 4.0, 7.0],
    "Fac U": [6.0, 7.0, 6.8, 6.2, 7.9, 6.7, 7.3, 6.6],
    "Qlt S": [6.7, 5.8, 6.2, 5.8, 6.6, 5.6, 5.8, 5.9],
    "Qlt Com": [8.8, 6.1, 7.8, 5.4, 7.0, 5.6, 5.6, 6.9],
    "Son": [2.1, 1.9, 2.0, 1.4, 2.0, 1.7, 1.7, 2.0]
}
df = pd.DataFrame(data).set_index("Marque")

# Standardisation (ACPN)
scaler = StandardScaler()
data_std = scaler.fit_transform(df)

# ACP avec 2 axes principaux
pca = PCA(n_components=2)
scores = pca.fit_transform(data_std)  # Coordonnées des marques
components = pca.components_         # Corrélations variables-axes

print("Variance expliquée par axe :", pca.explained_variance_ratio_)

# Corrélations variables-axes (pour le cercle de corrélations)
corr_var = components.T * np.sqrt(pca.explained_variance_)
print(pd.DataFrame(corr_var, index=df.columns, columns=["CP1", "CP2"]))

print(pd.DataFrame(scores, index=df.index, columns=["CP1", "CP2"]))


plt.figure(figsize=(10, 6))

# Projection des marques
plt.scatter(scores[:, 0], scores[:, 1], c='red', label='Marques')
for i, marque in enumerate(df.index):
    plt.text(scores[i, 0], scores[i, 1], marque, fontsize=12)

# Flèches des variables
for j, var in enumerate(df.columns):
    plt.arrow(0, 0, corr_var[j, 0], corr_var[j, 1], color='blue', alpha=0.7, head_width=0.1)
    plt.text(corr_var[j, 0]*1.1, corr_var[j, 1]*1.1, var, fontsize=12, color='darkblue')

# Paramètres
plt.axhline(0, color='grey', linestyle='--')
plt.axvline(0, color='grey', linestyle='--')
plt.xlabel(f"CP1 (45%)")
plt.ylabel(f"CP2 (30%)")
plt.title("Biplot ACPN : Marques et Caractéristiques")
plt.grid()
plt.show()