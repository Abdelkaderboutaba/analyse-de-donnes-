import numpy as np
import pandas as pd

# Déclaration de la matrice des données
X = np.array([
    [6, 6, 5, 5.5, 8],
    [8, 8, 8, 8, 9],
    [6, 7, 11, 9.5, 11],
    [14.5, 14.5, 15.5, 15, 8],
    [14, 14, 12, 12, 10],
    [11, 10, 5.5, 7, 13],
    [5.5, 7, 14, 11.5, 10],
    [13, 12.5, 8.5, 9.5, 12],
    [9, 9.5, 12.5, 12, 18]
])

# Transposée de X
X_t = X.T

# Affichage
# print("Matrice X :\n", X)
# print("\nTransposée Xᵗ :\n", X_t)

individus = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"]
modules = ["Multimedia", "Maths", "Système", "Réseau", "Autre"]


df = pd.DataFrame(X, index=individus, columns=modules)

# Affichage de la liste des individus
print("\nListe des individus :", individus)

# Extraction d'une variable (exemple : 'Maths')
maths = df["Maths"]
print("\nVariable Maths :\n", maths)

print("\nIndividus 4, 5 et 7 :\n", df.iloc[[3, 4, 6]])


# Calcul de la moyenne, variance et écart type pour chaque variable
stats = df.describe().T[['mean', 'std']]
stats['variance'] = df.var()
stats = stats.round(4)  # 4 chiffres après la virgule

print("\nStatistiques (moyenne, variance, écart type) :\n", stats)


# Calcul de la moyenne des notes de chaque étudiant
individu_moyen = df.mean(axis=1).round(4)

# Affichage
print("\nIndividu moyen :\n", individu_moyen)


def calcul_variance(data):
    return np.var(data, axis=0, ddof=1).round(4)

variances = calcul_variance(X)
print("\nVariance des 5 variables :\n", variances)


# Matrice de covariance
V = np.cov(X, rowvar=False)
V = np.round(V, 4)

print("\nMatrice des covariances :\n", V)


# Matrice de corrélation
R = np.corrcoef(X, rowvar=False)
R = np.round(R, 4)

print("\nMatrice des corrélations :\n", R)



import matplotlib.pyplot as plt

# Paires de variables à tracer
paires = [(0, 4), (1, 3), (2, 3)]
titles = [("Multimedia", "Autre"), ("Maths", "Réseau"), ("Système", "Réseau")]

plt.figure(figsize=(12, 4))

for i, (x_idx, y_idx) in enumerate(paires):
    plt.subplot(1, 3, i+1)
    plt.scatter(X[:, x_idx], X[:, y_idx], c='blue', alpha=0.7)
    plt.xlabel(titles[i][0])
    plt.ylabel(titles[i][1])
    plt.title(f"{titles[i][0]} vs {titles[i][1]}")

plt.tight_layout()
plt.show()
