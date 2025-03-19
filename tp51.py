import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Données du tableau
data = {
    'Marque': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'],
    'Auto': [7.5, 4.0, 6.9, 8.6, 6.3, 9.3, 4.0, 7.0],
    'Fac U': [6.0, 7.0, 6.8, 6.2, 7.9, 6.7, 7.3, 6.6],
    'Qlt S': [6.7, 5.8, 6.2, 5.8, 6.6, 5.6, 5.8, 5.9],
    'Qlt Com': [8.8, 6.1, 7.8, 5.4, 7.0, 5.6, 5.6, 6.9],
    'Son': [2.1, 1.9, 2.0, 1.4, 2.0, 1.7, 1.7, 2.0]
}

# Créer un DataFrame
df = pd.DataFrame(data)
df.set_index('Marque', inplace=True)

# Centrer et réduire les données
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Appliquer l'ACP
pca = PCA()
pca.fit(df_scaled)

# Obtenir les composantes principales
components = pca.transform(df_scaled)


# Créer un DataFrame pour les composantes principales
df_pca = pd.DataFrame(components, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], index=df.index)

# Visualiser les individus dans un sous-espace 2D
plt.figure(figsize=(8, 6))
plt.scatter(df_pca['PC1'], df_pca['PC2'], c='blue', label='Marques')
for i, txt in enumerate(df.index):
    plt.annotate(txt, (df_pca['PC1'][i], df_pca['PC2'][i]))
plt.xlabel('Composante Principale 1 (PC1)')
plt.ylabel('Composante Principale 2 (PC2)')
plt.title('Projection des marques sur les deux premières composantes principales')
plt.legend()
plt.grid()
plt.show()


# Obtenir les coefficients des variables sur les composantes principales
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Visualiser les variables dans un sous-espace 2D
plt.figure(figsize=(8, 6))
for i, feature in enumerate(df.columns):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='red', alpha=0.5)
    plt.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, feature, color='red', ha='center', va='center')
plt.xlabel('Composante Principale 1 (PC1)')
plt.ylabel('Composante Principale 2 (PC2)')
plt.title('Projection des variables sur les deux premières composantes principales')
plt.grid()
plt.show()


# Visualisation des loadings dans un espace 2D (PC1 vs PC2)
plt.figure(figsize=(8, 6))
for i, feature in enumerate(df.columns):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='red', alpha=0.5, head_width=0.05)
    plt.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, feature, color='red', ha='center', va='center')
plt.xlabel('Composante Principale 1 (PC1)')
plt.ylabel('Composante Principale 2 (PC2)')
plt.title('Projection des variables sur les deux premières composantes principales')
plt.grid()
plt.show()