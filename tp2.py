import numpy as np
from scipy.spatial.distance import euclidean

# Définition des individus
X = np.array([
    [6, 6, 5, 5.5, 8],  
    [8, 8, 8, 8, 9],  
    [6, 7, 11, 9.5, 11],  
    [14.5, 14.5, 15.5, 15, 8],  # E4
    [14, 14, 12, 12, 10],  # E5
    [11, 10, 5.5, 7, 13],  
    [5.5, 7, 14, 11.5, 10],  # E7
    [13, 12.5, 8.5, 9.5, 12],  
    [9, 9.5, 12.5, 12, 18]
])

# Sélection des individus
E4, E5, E7 = X[3], X[4], X[6]

# Calcul des distances euclidiennes
dist_4_5 = euclidean(E4, E5)
dist_4_7 = euclidean(E4, E7)
dist_5_7 = euclidean(E5, E7)

# Affichage des résultats
print(f"Distance entre E4 et E5 : {dist_4_5:.4f}")
print(f"Distance entre E4 et E7 : {dist_4_7:.4f}")
print(f"Distance entre E5 et E7 : {dist_5_7:.4f}")


from sklearn.preprocessing import StandardScaler

# Normalisation des données (centrage-réduction)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sélection des individus normalisés
E4_scaled, E5_scaled, E7_scaled = X_scaled[3], X_scaled[4], X_scaled[6]

# Recalcul des distances après standardisation
dist_4_5_scaled = euclidean(E4_scaled, E5_scaled)
dist_4_7_scaled = euclidean(E4_scaled, E7_scaled)
dist_5_7_scaled = euclidean(E5_scaled, E7_scaled)

print(f"Distance normalisée entre E4 et E5 : {dist_4_5_scaled:.4f}")
print(f"Distance normalisée entre E4 et E7 : {dist_4_7_scaled:.4f}")
print(f"Distance normalisée entre E5 et E7 : {dist_5_7_scaled:.4f}")




from numpy.linalg import eig

# Calcul de la matrice de covariance
cov_matrix = np.cov(X_scaled, rowvar=False)

# Calcul des valeurs propres et vecteurs propres
valeurs_propres, vecteurs_propres = eig(cov_matrix)

# Affichage des résultats
print("\nValeurs propres :\n", valeurs_propres.round(4))
print("\nVecteurs propres :\n", vecteurs_propres.round(4))



