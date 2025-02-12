import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Charger un jeu de données
iris = load_iris()
X = iris.data  # Variables initiales

# Appliquer l'ACP pour réduire à 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Affichage des données transformées
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap='viridis', alpha=0.7)
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.title("ACP sur le jeu de données Iris")
plt.colorbar()
plt.show()
