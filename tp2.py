import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Liste des individus (étudiants)
individus = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"]

# Fonction pour calculer la distance euclidienne
def distance_euclidienne(v1, v2):
    return np.linalg.norm(v1 - v2)

# Distances Euclidiennes sur les données brutes
dist_E4_E5 = distance_euclidienne(X[3], X[4])
dist_E4_E7 = distance_euclidienne(X[3], X[6])
dist_E5_E7 = distance_euclidienne(X[4], X[6])

print("Distances Euclidiennes sur les données brutes:")
print(f"E4 - E5 : {dist_E4_E5:.4f}")
print(f"E4 - E7 : {dist_E4_E7:.4f}")
print(f"E5 - E7 : {dist_E5_E7:.4f}")

# Centrage-réduction des données (standardisation)
moyennes = np.mean(X, axis=0)
ecarts_types = np.std(X, axis=0, ddof=0)
X_standard = (X - moyennes) / ecarts_types

# Distances Euclidiennes sur les données centrées-réduites
dist_E4_E5_std = distance_euclidienne(X_standard[3], X_standard[4])
dist_E4_E7_std = distance_euclidienne(X_standard[3], X_standard[6])
dist_E5_E7_std = distance_euclidienne(X_standard[4], X_standard[6])

print("\nDistances Euclidiennes sur les données centrées-réduites:")
print(f"E4 - E5 : {dist_E4_E5_std:.4f}")
print(f"E4 - E7 : {dist_E4_E7_std:.4f}")
print(f"E5 - E7 : {dist_E5_E7_std:.4f}")



# ✅ E4 et E5 sont proches → Ils ont des notes similaires.
# ❌ E7 est plus éloigné de E4 et E5 → Son profil est différent.
# 📏 La standardisation a ajusté l’impact des valeurs élevées, mais n’a pas changé la tendance générale.


# --- Analyse en Composantes Principales (ACP) ---
# Calcul de la matrice de corrélation
matrice_correlation = np.corrcoef(X_standard, rowvar=False)

# Calcul des valeurs propres et vecteurs propres
valeurs_propres, vecteurs_propres = np.linalg.eig(matrice_correlation)

# Affichage des résultats
print("\nValeurs propres:")
print(valeurs_propres)
print("\nVecteurs propres:")
print(vecteurs_propres)

def est_vecteur_propre(A, v, lambda_):
    Av = np.dot(A, v)
    return np.allclose(Av, lambda_ * v)

# Vérification des vecteurs propres
for i in range(len(valeurs_propres)):
    lambda_i = valeurs_propres[i]
    v_i = vecteurs_propres[:, i]
    resultat = est_vecteur_propre(matrice_correlation, v_i, lambda_i)
    print(f"Vecteur propre {i+1} vérification: {resultat}")

































































# import numpy as np
# import pandas as pd

# # Normalisation des données (centrage-réduction)
# from sklearn.preprocessing import StandardScaler

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

# # Calcul de la matrice de corrélation
# correlation_matrix = np.corrcoef(X_scaled, rowvar=False)

# # Affichage sous forme de DataFrame
# df_corr = pd.DataFrame(correlation_matrix, columns=["Multimedia", "Maths", "Système", "Réseau", "Autre"], 
#                        index=["Multimedia", "Maths", "Système", "Réseau", "Autre"])

# print("\nMatrice de corrélation :\n", df_corr.round(4))


# from numpy.linalg import eig

# # Calcul des valeurs propres et vecteurs propres
# valeurs_propres, vecteurs_propres = eig(correlation_matrix)

# # Affichage des résultats
# print("\nValeurs propres :\n", valeurs_propres.round(4))
# print("\nVecteurs propres :\n", vecteurs_propres.round(4))


# # Vérification de la relation Av = λv
# A = correlation_matrix  # Matrice utilisée
# v = vecteurs_propres[:, 0]  # Premier vecteur propre
# λ = valeurs_propres[0]  # Première valeur propre

# # Vérification
# test = np.dot(A, v)
# expected = λ * v

# # Vérification
# print("\nLe vecteur est-il bien un vecteur propre ?", np.allclose(test, expected))



# import matplotlib.pyplot as plt

# # Tracer les valeurs propres
# plt.figure(figsize=(8, 5))
# plt.bar(range(1, len(valeurs_propres) + 1), valeurs_propres, color='blue', alpha=0.7)
# plt.xlabel("Numéro de la composante")
# plt.ylabel("Valeur propre")
# plt.title("Distribution des valeurs propres")
# plt.show()


# # Calcul des taux d'inertie
# taux_inertie = valeurs_propres / sum(valeurs_propres)
# taux_inertie_cumulé = np.cumsum(taux_inertie)

# # Création du tableau
# df_inertie = pd.DataFrame({
#     "Valeurs propres": valeurs_propres.round(4),
#     "Taux d'inertie (%)": (taux_inertie * 100).round(2),
#     "Taux d'inertie cumulé (%)": (taux_inertie_cumulé * 100).round(2)
# })

# print("\nTableau des valeurs propres et inerties :\n", df_inertie)


# # Projection des individus sur les axes principaux
# X_projected = np.dot(X_scaled, vecteurs_propres)

# # Affichage des projections
# df_projections = pd.DataFrame(X_projected[:, :2], index=["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"], columns=["PC1", "PC2"])
# print("\nProjections des individus :\n", df_projections)



# import matplotlib.pyplot as plt

# # Création du graphique
# plt.figure(figsize=(8, 6))

# # Affichage des points
# plt.scatter(X_projected[:, 0], X_projected[:, 1], c='blue', alpha=0.7)

# # Ajout des labels (noms des individus)
# for i, individu in enumerate(["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"]):
#     plt.text(X_projected[i, 0], X_projected[i, 1], individu, fontsize=12, ha='right')

# # Ajout des axes et titre
# plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Axe horizontal
# plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)  # Axe vertical
# plt.xlabel("Composante Principale 1 (PC1)")
# plt.ylabel("Composante Principale 2 (PC2)")
# plt.title("Projection des Individus dans le Nouveau Sous-Espace")
# plt.grid()

# # Affichage du graphique
# plt.show()


