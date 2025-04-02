
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Données fournies (matrice 9x5)
data = np.array([
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

# 1. Standardisation des données
scaler = StandardScaler()
data_std = scaler.fit_transform(data)

# 2. ACP
pca = PCA()
scores = pca.fit_transform(data_std)  # Coordonnées factorielles


# 3. Carré de la distance au centre (origine dans le sous-espace factoriel)
squared_distances = np.sum(scores**2, axis=1)  # Somme des carrés des coordonnées
print(squared_distances)

# Affichage des résultats
for i, dist in enumerate(squared_distances):
    print(f"Individu E_{i+1} : d² = {dist:.4f}")

# Optionnel : Variance expliquée par chaque axe
print("\nVariance expliquée par chaque composante :", pca.explained_variance_ratio_)



































































# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from scipy.spatial.distance import euclidean

# # ----------------------
# # 1️⃣ Chargement des données et ACP (Résultats du TP2)
# # ----------------------
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

# # Standardisation (centrage-réduction)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Matrice de corrélation
# correlation_matrix = np.corrcoef(X_scaled, rowvar=False)

# # Valeurs propres et vecteurs propres
# valeurs_propres, vecteurs_propres = np.linalg.eig(correlation_matrix)

# # Projection des individus
# X_projected = np.dot(X_scaled, vecteurs_propres)

# # ----------------------
# # 2️⃣ Distance des individus au centre du nuage
# # ----------------------
# dist_carre = np.sum(X_projected[:, :3] ** 2, axis=1)
# df_distance = pd.DataFrame({
#     "Individu": ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"],
#     "Distance² au centre": dist_carre.round(4)
# })
# print("\nCarré de la distance des individus au centre du nuage :\n", df_distance)


# #✅ Interprétation :

# # Les grandes distances indiquent des individus atypiques.
# # Une distance faible signifie que l’individu est moyen par rapport aux autres.






# # ----------------------
# # 3️⃣ Qualité de représentation (COS²)
# # ----------------------
# qualite_representation = (X_projected[:, :3] ** 2) / dist_carre[:, np.newaxis]
# df_qualite = pd.DataFrame(qualite_representation.round(4), 
#                           index=["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"], 
#                           columns=["PC1", "PC2", "PC3"])
# print("\nQualité de représentation des individus (COS²) :\n", df_qualite)


# # ✅ Interprétation :

# # Si un individu a un COS² faible sur PC1 et PC2, il est mieux représenté sur PC3 ou d’autres axes.
# # Si un individu a COS² ≈ 1 sur PC1, il est principalement caractérisé par cet axe.







# # ----------------------
# # 4️⃣ Contribution des individus à l’inertie des axes
# # ----------------------
# contribution = (X_projected[:, :3] ** 2) / (X.shape[0] * valeurs_propres[:3])
# df_contribution = pd.DataFrame(contribution.round(4), 
#                                index=["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"], 
#                                columns=["PC1", "PC2", "PC3"])
# print("\nContribution des individus à l’inertie des axes :\n", df_contribution)



# # ✅ Interprétation :

# # Si un individu a une forte contribution sur un axe, il est très influent.
# # Un individu avec une faible contribution sur tous les axes est neutre/moyen.






# # ----------------------
# # 5️⃣ Coordonnées des variables sur les axes principaux
# # ----------------------
# coord_var = vecteurs_propres[:, :3] * np.sqrt(valeurs_propres[:3])
# df_coord_var = pd.DataFrame(coord_var.round(4), 
#                             index=["Multimedia", "Maths", "Système", "Réseau", "Autre"], 
#                             columns=["PC1", "PC2", "PC3"])
# print("\nCoordonnées des variables sur les axes principaux :\n", df_coord_var)


# # ✅ Interprétation :

# # Une variable proche de 1 sur PC1 est fortement corrélée avec PC1.
# # Si une variable a une forte valeur sur PC2 et une faible sur PC1, alors elle est mieux représentée par PC2.




# # ----------------------
# # 6️⃣ Cercle des corrélations
# # ----------------------
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# # Cercle PC1 vs PC2
# axes[0].scatter(df_coord_var["PC1"], df_coord_var["PC2"], c='red')
# for i, var in enumerate(df_coord_var.index):
#     axes[0].text(df_coord_var.iloc[i, 0], df_coord_var.iloc[i, 1], var, fontsize=12)
# axes[0].axhline(0, color='gray', linestyle='--')
# axes[0].axvline(0, color='gray', linestyle='--')
# axes[0].set_xlabel("PC1")
# axes[0].set_ylabel("PC2")
# axes[0].set_title("Cercle des Corrélations (PC1 vs PC2)")

# # Cercle PC1 vs PC3
# axes[1].scatter(df_coord_var["PC1"], df_coord_var["PC3"], c='blue')
# for i, var in enumerate(df_coord_var.index):
#     axes[1].text(df_coord_var.iloc[i, 0], df_coord_var.iloc[i, 2], var, fontsize=12)
# axes[1].axhline(0, color='gray', linestyle='--')
# axes[1].axvline(0, color='gray', linestyle='--')
# axes[1].set_xlabel("PC1")
# axes[1].set_ylabel("PC3")
# axes[1].set_title("Cercle des Corrélations (PC1 vs PC3)")

# plt.tight_layout()
# plt.show()


# # ✅ Interprétation :

# # Si deux variables sont proches → Elles sont corrélées.
# # Si une variable est proche de (1,0) sur PC1, elle est fortement liée à cet axe.





# # ----------------------
# # 7️⃣ Biplot : Projection des individus et variables
# # ----------------------
# plt.figure(figsize=(8, 6))

# # Tracer les individus
# plt.scatter(X_projected[:, 0], X_projected[:, 1], c='blue', alpha=0.7)
# for i, individu in enumerate(["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"]):
#     plt.text(X_projected[i, 0], X_projected[i, 1], individu, fontsize=10, color='blue')

# # Tracer les variables
# for i, var in enumerate(df_coord_var.index):
#     plt.arrow(0, 0, df_coord_var.iloc[i, 0], df_coord_var.iloc[i, 1], color='red', alpha=0.5)
#     plt.text(df_coord_var.iloc[i, 0], df_coord_var.iloc[i, 1], var, fontsize=12, color='red')

# plt.axhline(0, color='gray', linestyle='--')
# plt.axvline(0, color='gray', linestyle='--')
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("Biplot : Indices et Variables")
# plt.grid()
# plt.show()



# # ✅ Interprétation :

# # Les individus proches d’une variable sont associés à cette variable.
# # Les flèches rouges montrent la relation entre variables et axes.