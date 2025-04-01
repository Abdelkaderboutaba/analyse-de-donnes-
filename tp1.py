import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

# Transposée de la matrice
X_t = X.T

# Affichage des matrices
print("Matrice X :\n", X)
print("\nMatrice Transposée X_t :\n", X_t)

# Liste des individus (étudiants)
individus = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"]
print("\nListe des individus :", individus)

# Extraction des variables (modules)
modules = ["Multimedia", "Maths", "Système", "Réseau", "Autre"]
variables = X.T  # Chaque colonne représente une variable
print("\nVariables extraites :\n", variables)

# Création du tableau X(j) avec moyenne, variance et écart-type
moyennes = np.round(np.mean(X, axis=0), 4)
variances = np.round(np.var(X, axis=0, ddof=0), 4)
ecart_types = np.round(np.std(X, axis=0, ddof=0), 4)

# Création du DataFrame
X_j = pd.DataFrame({
    'Moyenne': moyennes,
    'Variance': variances,
    'Écart-type': ecart_types
}, index=modules)

# Affichage du tableau X(j)
print("\nTableau X(j) :\n", X_j)

# Calcul de l'individu moyen
individu_moyen = np.round(np.mean(X, axis=1), 4)

# Création du DataFrame pour affichage
individu_moyen_df = pd.DataFrame({'Individu Moyen': individu_moyen}, index=individus)

# Affichage de l'individu moyen
print("\nIndividu Moyen :\n", individu_moyen_df)

# Fonction pour calculer la variance des 5 variables
def calculer_variance(X):
    return np.round(np.var(X, axis=0, ddof=0), 4)

# Calcul et affichage des variances
variances_calculées = calculer_variance(X)
print("\nVariances des 5 variables :", variances_calculées)

# Calcul de la matrice de covariance
V = np.round(np.cov(X, rowvar=False, ddof=0), 4)

# Création du DataFrame pour affichage
V_df = pd.DataFrame(V, index=modules, columns=modules)

# Affichage de la matrice de covariance
print("\nMatrice de covariance V :\n", V_df)


# Calcul de la matrice de corrélation
R = np.round(np.corrcoef(X, rowvar=False), 4)

# Création du DataFrame pour affichage
R_df = pd.DataFrame(R, index=modules, columns=modules)

# Affichage de la matrice de corrélation
print("\nMatrice de corrélation R :\n", R_df)

# Analyse et interprétation des résultats de la matrice de covariance
print("\nInterprétation de la matrice de covariance :")
for i in range(len(modules)):
    for j in range(i+1, len(modules)):
        print(f"La covariance entre {modules[i]} et {modules[j]} est de {V[i, j]}")
        if V[i, j] > 0:
            print(" → Ces variables ont une relation positive : elles augmentent ou diminuent ensemble.")
        elif V[i, j] < 0:
            print(" → Ces variables ont une relation négative : lorsque l'une augmente, l'autre diminue.")
        else:
            print(" → Ces variables ne sont pas corrélées.")

# Affichage d'une heatmap pour la matrice de covariance
plt.figure(figsize=(8, 6))
sns.heatmap(V_df, annot=True, cmap="coolwarm", fmt=".4f", linewidths=0.5)
plt.title("Matrice de covariance")
plt.show()

# Affichage d'une heatmap pour la matrice de corrélation
plt.figure(figsize=(8, 6))
sns.heatmap(R_df, annot=True, cmap="coolwarm", fmt=".4f", linewidths=0.5)
plt.title("Matrice de corrélation")
plt.show()


# Représentation graphique des individus dans l’espace ℜ2
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

couples = [(0, 4), (1, 3), (2, 3)]
noms_couples = [("Multimedia", "Autre"), ("Maths", "Réseau"), ("Système", "Réseau")]

for i, (x_idx, y_idx) in enumerate(couples):
    axs[i].scatter(X[:, x_idx], X[:, y_idx], color='blue', alpha=0.7)
    axs[i].set_xlabel(noms_couples[i][0])
    axs[i].set_ylabel(noms_couples[i][1])
    axs[i].set_title(f"Nuage de points : {noms_couples[i][0]} vs {noms_couples[i][1]}")

plt.tight_layout()
plt.show()

































































































# import numpy as np
# import pandas as pd

# # Déclaration de la matrice des données
# X = np.array([
#     [6, 6, 5, 5.5, 8],
#     [8, 8, 8, 8, 9],
#     [6, 7, 11, 9.5, 11],
#     [14.5, 14.5, 15.5, 15, 8],
#     [14, 14, 12, 12, 10],
#     [11, 10, 5.5, 7, 13],
#     [5.5, 7, 14, 11.5, 10],
#     [13, 12.5, 8.5, 9.5, 12],
#     [9, 9.5, 12.5, 12, 18]
# ])

# # Transposée de X
# X_t = X.T

# # Affichage
# # print("Matrice X :\n", X)
# # print("\nTransposée Xᵗ :\n", X_t)

# individus = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"]
# modules = ["Multimedia", "Maths", "Système", "Réseau", "Autre"]


# df = pd.DataFrame(X, index=individus, columns=modules)

# # Affichage de la liste des individus
# print("\nListe des individus :", individus)

# # Extraction d'une variable (exemple : 'Maths')
# maths = df["Maths"]
# print("\nVariable Maths :\n", maths)

# print("\nIndividus 4, 5 et 7 :\n", df.iloc[[3, 4, 6]])


# # Calcul de la moyenne, variance et écart type pour chaque variable
# stats = df.describe().T[['mean', 'std']]
# stats['variance'] = df.var()
# stats = stats.round(4)  # 4 chiffres après la virgule

# print("\nStatistiques (moyenne, variance, écart type) :\n", stats)


# # Calcul de la moyenne des notes de chaque étudiant
# individu_moyen = df.mean(axis=1).round(4)

# # Affichage
# print("\nIndividu moyen :\n", individu_moyen)


# def calcul_variance(data):
#     return np.var(data, axis=0, ddof=1).round(4)

# variances = calcul_variance(X)
# print("\nVariance des 5 variables :\n", variances)


# # Matrice de covariance
# V = np.cov(X, rowvar=False)
# V = np.round(V, 4)

# print("\nMatrice des covariances :\n", V)


# # Matrice de corrélation
# R = np.corrcoef(X, rowvar=False)
# R = np.round(R, 4)

# print("\nMatrice des corrélations :\n", R)



# import matplotlib.pyplot as plt

# # Paires de variables à tracer
# paires = [(0, 4), (1, 3), (2, 3)]
# titles = [("Multimedia", "Autre"), ("Maths", "Réseau"), ("Système", "Réseau")]

# plt.figure(figsize=(12, 4))

# for i, (x_idx, y_idx) in enumerate(paires):
#     plt.subplot(1, 3, i+1)
#     plt.scatter(X[:, x_idx], X[:, y_idx], c='blue', alpha=0.7)
#     plt.xlabel(titles[i][0])
#     plt.ylabel(titles[i][1])
#     plt.title(f"{titles[i][0]} vs {titles[i][1]}")

# plt.tight_layout()
# plt.show()
