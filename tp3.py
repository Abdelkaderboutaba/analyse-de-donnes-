
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
variables = ["Mul", "Maths", "Système", "Réseau", "Autre"]
individus = [f"E{i+1}" for i in range(9)]
# 1. Standardisation des données
scaler = StandardScaler()
data_std = scaler.fit_transform(data)

# 2. ACP avec 3 composantes principales
pca = PCA(n_components=3)
scores = pca.fit_transform(data_std)  # Coordonnées factorielles (individus x 3 CP)
coords_vars = pca.components_.T * np.sqrt(pca.explained_variance_)


# 3. Calcul des cos² (qualité de représentation)
## Étape 1 : Carré des distances dans l'espace complet (toutes CP)
d2_full = np.sum(scores**2, axis=1)  # d² = somme des carrés des coordonnées sur toutes les CP

## Étape 2 : Cos² par axe (pour chaque individu et chaque CP)
cos2 = (scores**2) / d2_full.reshape(-1, 1)  # Normalisation par d²

# 4. Création d'un tableau de résultats (DataFrame pandas)
results = pd.DataFrame(
    cos2,
    columns=[f"Axe {k+1} (Cos²)" for k in range(3)],
    index=[f"E{i+1}" for i in range(9)]
)

# Ajout de la qualité globale dans le plan factoriel (Axe 1 + Axe 2)
results["Plan (Axe1+Axe2)"] = results.iloc[:, 0] + results.iloc[:, 1]

# Affichage avec 2 décimales
print("Qualité de représentation des individus (Cos²) :")
print(results.round(2))

# Optionnel : Variance expliquée par chaque axe
print("\nVariance expliquée par les axes :")
print(pca.explained_variance_ratio_.round(4))



# Projection dans le plan factoriel (Axe 1 vs Axe 2)
plt.figure(figsize=(10, 6))
plt.scatter(scores[:, 0], scores[:, 1], c='blue', s=100)
for i, txt in enumerate([f"E{i+1}" for i in range(9)]):
    plt.annotate(txt, (scores[i, 0], scores[i, 1]), fontsize=12)
plt.xlabel("Axe 1 ({}% variance)".format(pca.explained_variance_ratio_[0]*100))
plt.ylabel("Axe 2 ({}% variance)".format(pca.explained_variance_ratio_[1]*100))
plt.title("Projection des Individus dans le Plan Factoriel")
plt.grid()
plt.show()



inertie_totale = data_std.shape[0]  # = 9

## Étape 2 : Contribution relative de chaque individu à chaque axe
contributions = (scores**2) / (inertie_totale * pca.explained_variance_) 

# 4. Création d'un tableau de résultats (DataFrame pandas)
results = pd.DataFrame(
    contributions * 100,  # Convertir en pourcentage
    columns=[f"Axe {k+1} (%)" for k in range(3)],
    index=[f"E{i+1}" for i in range(9)]
)

# Affichage avec 2 décimales
print("Contribution des individus à l'inertie de chaque axe (%):")
print(results.round(2))

# Optionnel : Variance expliquée par chaque axe
print("\nVariance expliquée par les axes :")
print(pca.explained_variance_ratio_.round(4))

# Seuil de mauvaise représentation (cos² < 0.5)
seuil = 0.5

# DataFrame pour les résultats
results = pd.DataFrame({
    "Axe 1": ["✓" if cos2[i, 0] >= seuil else f"Mal (signe: {'+' if scores[i, 0] >= 0 else '-'})" for i in range(9)],
    "Axe 2": ["✓" if cos2[i, 1] >= seuil else f"Mal (signe: {'+' if scores[i, 1] >= 0 else '-'})" for i in range(9)],
    "Axe 3": ["✓" if cos2[i, 2] >= seuil else f"Mal (signe: {'+' if scores[i, 2] >= 0 else '-'})" for i in range(9)]
}, index=[f"E{i+1}" for i in range(9)])

print("Individus mal représentés par axe (cos² < 0.5) :")
print(results)



plt.figure(figsize=(10, 6))
plt.scatter(scores[:, 0], scores[:, 1], c='blue', s=100)
for i, txt in enumerate([f"E{i+1}" for i in range(9)]):
    plt.annotate(txt, (scores[i, 0], scores[i, 1]), fontsize=12)
plt.xlabel("Axe 1")
plt.ylabel("Axe 2")
plt.title("Plan Factoriel (Axe 1 vs Axe 2)")
plt.grid()
plt.show()


# 3. Coordonnées des variables (matrice de corrélations variables-axes)
coords_variables = pca.components_.T * np.sqrt(pca.explained_variance_)

# 4. Création d'un tableau de résultats
results = pd.DataFrame(
    coords_variables,
    columns=[f"CP{k+1}" for k in range(3)],
    index=variables
)

# Affichage
print("Coordonnées des variables sur les axes principaux :")
print(results.round(4))

# Optionnel : Variance expliquée
print("\nVariance expliquée par chaque axe :")
print(pca.explained_variance_ratio_.round(4))


fig, ax = plt.subplots(figsize=(10, 8))
# Axes 1 et 2
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.axhline(0, color='grey', linestyle='--')
ax.axvline(0, color='grey', linestyle='--')
ax.add_patch(plt.Circle((0, 0), 1, color='blue', fill=False))

# Flèches pour les variables
for i, var in enumerate(variables):
    ax.arrow(0, 0, coords_variables[i, 0], coords_variables[i, 1], 
             head_width=0.05, color='red')
    ax.text(coords_variables[i, 0]*1.1, coords_variables[i, 1]*1.1, var, fontsize=12)

ax.set_xlabel("CP1 ({}%)".format(pca.explained_variance_ratio_[0]*100))
ax.set_ylabel("CP2 ({}%)".format(pca.explained_variance_ratio_[1]*100))
ax.set_title("Cercle des Corrélations (CP1 vs CP2)")
plt.grid()
plt.show()


# Taux d'inertie (en %)
inertie = pca.explained_variance_ratio_ * 100

# Création de la figure avec 2 sous-graphiques
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# ---- Cercle CP1 vs CP2 ----
ax1.set_xlim(-1.1, 1.1)
ax1.set_ylim(-1.1, 1.1)
ax1.axhline(0, color='grey', alpha=0.6)
ax1.axvline(0, color='grey', alpha=0.6)
ax1.add_patch(plt.Circle((0, 0), 1, color='blue', fill=False, alpha=0.3))

# Flèches et annotations
for i, var in enumerate(variables):
    ax1.arrow(0, 0, coords_vars[i, 0], coords_vars[i, 1], 
              head_width=0.05, color='red', alpha=0.7)
    ax1.text(coords_vars[i, 0]*1.1, coords_vars[i, 1]*1.1, var, 
             fontsize=12, color='darkred')

ax1.set_xlabel(f"CP1 ({inertie[0]:.1f}%)", fontsize=12)
ax1.set_ylabel(f"CP2 ({inertie[1]:.1f}%)", fontsize=12)
ax1.set_title("Cercle des Corrélations (CP1 vs CP2)", fontsize=14)
ax1.grid(alpha=0.3)

# ---- Cercle CP1 vs CP3 ----
ax2.set_xlim(-1.1, 1.1)
ax2.set_ylim(-1.1, 1.1)
ax2.axhline(0, color='grey', alpha=0.6)
ax2.axvline(0, color='grey', alpha=0.6)
ax2.add_patch(plt.Circle((0, 0), 1, color='blue', fill=False, alpha=0.3))


for i, var in enumerate(variables):
    ax2.arrow(0, 0, coords_vars[i, 0], coords_vars[i, 2], 
              head_width=0.05, color='green', alpha=0.7)
    ax2.text(coords_vars[i, 0]*1.1, coords_vars[i, 2]*1.1, var, 
             fontsize=12, color='darkgreen')

ax2.set_xlabel(f"CP1 ({inertie[0]:.1f}%)", fontsize=12)
ax2.set_ylabel(f"CP3 ({inertie[2]:.1f}%)", fontsize=12)
ax2.set_title("Cercle des Corrélations (CP1 vs CP3)", fontsize=14)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()


corr_var_axes = pca.components_.T * np.sqrt(pca.explained_variance_)

# Calcul des cos²
cos2 = (corr_var_axes ** 2) / np.sum(corr_var_axes ** 2, axis=1, keepdims=True)

# Tableau des résultats
results = pd.DataFrame(
    cos2 * 100,  # Convertir en pourcentage
    columns=[f"Axe {k+1} (cos² %)" for k in range(3)],
    index=variables
)

# Ajout de la qualité dans le plan factoriel (Axe1 + Axe2)
results["Plan (Axe1+Axe2)"] = results.iloc[:, 0] + results.iloc[:, 1]

print("Qualité de représentation des variables (cos² en %) :")
print(results.round(2))

contributions = (corr_var_axes ** 2) / pca.explained_variance_ * 100

# 3. Tableau des résultats
results = pd.DataFrame(
    contributions,
    columns=[f"Axe {k+1} (%)" for k in range(3)],
    index=variables
)

# Affichage
print("Contribution des variables à l'inertie de chaque axe (%):")
print(results.round(2))

# Optionnel : Variance expliquée par axe
print("\nVariance expliquée par les axes :")
print(pca.explained_variance_ratio_.round(4))

corr_var = pca.components_.T * np.sqrt(pca.explained_variance_)  # Coordonnées des variables

# Création du biplot
plt.figure(figsize=(12, 8))

# 1. Projection des individus (rouge)
plt.scatter(scores[:, 0], scores[:, 1], c='red', label='Individus', s=100)
for i, txt in enumerate(individus):
    plt.annotate(txt, (scores[i, 0], scores[i, 1]), fontsize=12, color='darkred')

# 2. Projection des variables (bleu)
for j, var in enumerate(variables):
    plt.arrow(0, 0, corr_var[j, 0], corr_var[j, 1], color='blue', alpha=0.7, head_width=0.1)
    plt.text(corr_var[j, 0]*1.1, corr_var[j, 1]*1.1, var, fontsize=12, color='darkblue')

# Paramètres du graphique
plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
plt.axvline(0, color='grey', linestyle='--', alpha=0.5)
plt.xlabel(f"CP1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
plt.ylabel(f"CP2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
plt.title("Biplot (Carte Factorielle) : Individus vs Variables", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.show()









































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