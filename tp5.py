import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.linalg import eig

# ----------------------
# 1️⃣ Chargement des données et standardisation
# ----------------------
data = {
    "Marque": ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"],
    "Auto": [7.5, 4.0, 6.9, 8.6, 6.3, 9.3, 4.0, 7.0],
    "Fac U": [6.0, 7.0, 6.8, 6.2, 7.9, 6.7, 7.3, 6.6],
    "Qlt S": [6.7, 5.8, 6.2, 5.8, 6.6, 5.6, 5.8, 5.9],
    "Qlt Com": [8.8, 6.1, 7.8, 5.4, 7.0, 5.6, 5.6, 6.9],
    "Son": [2.1, 1.9, 2.0, 1.4, 2.0, 1.7, 1.7, 2.0]
}

# Création du DataFrame
df = pd.DataFrame(data)
marques = df["Marque"]
df = df.drop(columns=["Marque"])

# Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ----------------------
# 2️⃣ Calcul de l’ACPN (matrice de corrélation, valeurs propres, vecteurs propres)
# ----------------------
correlation_matrix = np.corrcoef(X_scaled, rowvar=False)
valeurs_propres, vecteurs_propres = eig(correlation_matrix)

# Projection des individus
X_projected = np.dot(X_scaled, vecteurs_propres)

# ----------------------
# 3️⃣ Visualisation des individus dans un sous-espace 2D (PC1 vs PC2)
# ----------------------
plt.figure(figsize=(8, 6))
plt.scatter(X_projected[:, 0], X_projected[:, 1], c='blue', alpha=0.7)
for i, marque in enumerate(marques):
    plt.text(X_projected[i, 0], X_projected[i, 1], marque, fontsize=12, ha='right')
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Projection des Marques dans le Nouveau Sous-Espace")
plt.grid()
plt.show()

# ----------------------
# 4️⃣ Visualisation des variables dans un sous-espace 2D (Cercle des corrélations)
# ----------------------
coord_var = vecteurs_propres[:, :2] * np.sqrt(valeurs_propres[:2])

plt.figure(figsize=(8, 6))
plt.scatter(coord_var[:, 0], coord_var[:, 1], c='red')
for i, var in enumerate(df.columns):
    plt.arrow(0, 0, coord_var[i, 0], coord_var[i, 1], color='red', alpha=0.5)
    plt.text(coord_var[i, 0], coord_var[i, 1], var, fontsize=12, color='red')
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Cercle des Corrélations")
plt.grid()
plt.show()

# ----------------------
# 5️⃣ Résumé des résultats et interprétation
# ----------------------
df_valeurs_propres = pd.DataFrame({
    "Valeurs propres": valeurs_propres.round(4),
    "Taux d'inertie (%)": (valeurs_propres / sum(valeurs_propres) * 100).round(2)
})
print("\nValeurs propres et taux d'inertie :\n", df_valeurs_propres)
