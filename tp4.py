import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ----------------------
# 1️⃣ Chargement des données et préparation
# ----------------------
data = {
    "Ville": ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20"],
    "F. Ball": [1881.9, 3369.8, 4467.4, 1862.1, 3499.8, 3903.2, 2620.7, 3678.4, 3840.5, 2170.2, 3920.4, 2599.6, 2828.5, 2498.7, 2685.1, 2739.3, 1662.1, 2469.9, 2350.7, 3177.7],
    "Natation": [96.8, 96.8, 138.2, 83.2, 287.0, 170.7, 129.5, 157.0, 187.9, 140.5, 128.0, 39.6, 211.3, 123.2, 41.2, 100.7, 81.1, 142.9, 38.7, 292.1],
    "Tennis": [14.2, 10.8, 9.5, 8.8, 11.5, 6.3, 4.2, 6.0, 10.2, 11.7, 7.2, 5.5, 9.9, 7.4, 2.3, 6.6, 10.1, 15.5, 2.4, 8.0],
    "Gym": [25.2, 51.6, 34.2, 27.6, 49.4, 42.0, 16.8, 24.9, 39.6, 31.1, 25.5, 19.4, 21.8, 26.5, 10.6, 22.0, 19.1, 30.9, 13.5, 34.8],
    "B. Ball": [1135.5, 1331.7, 2346.1, 972.6, 2139.4, 1935.2, 1346.0, 1682.6, 1859.9, 1351.1, 1911.5, 1050.8, 1085.0, 1086.2, 812.5, 1270.4, 872.2, 1165.5, 1253.1, 1400.0],
    "H. Ball": [278.3, 284.0, 312.3, 203.4, 358.0, 292.9, 131.8, 194.2, 449.1, 256.5, 64.1, 172.5, 209.0, 153.5, 89.8, 180.5, 123.3, 335.5, 170.0, 358.9]
}

df = pd.DataFrame(data).set_index("Ville")

# Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Calcul de la matrice de corrélation
correlation_matrix = np.corrcoef(X_scaled, rowvar=False)

# Valeurs propres et vecteurs propres
valeurs_propres, vecteurs_propres = np.linalg.eig(correlation_matrix)

# Projection des données
X_projected = np.dot(X_scaled, vecteurs_propres)

# ----------------------
# 2️⃣ Fonctions pour analyse des résultats
# ----------------------

def afficher_distance_carre():
    """ Calcule et affiche la distance au centre du nuage. """
    dist_carre = np.sum(X_projected[:, :3] ** 2, axis=1)
    df_distance = pd.DataFrame({"Distance² au centre": dist_carre.round(4)}, index=df.index)
    print("\n📌 Carré de la distance des villes au centre du nuage :\n", df_distance)

def afficher_cos2():
    """ Calcule et affiche la qualité de représentation des individus. """
    dist_carre = np.sum(X_projected[:, :3] ** 2, axis=1)
    qualite_representation = (X_projected[:, :3] ** 2) / dist_carre[:, np.newaxis]
    df_qualite = pd.DataFrame(qualite_representation.round(4), index=df.index, columns=["PC1", "PC2", "PC3"])
    print("\n📌 Qualité de représentation des villes (COS²) :\n", df_qualite)

def afficher_contribution():
    """ Calcule et affiche la contribution des villes à l’inertie des axes. """
    contribution = (X_projected[:, :3] ** 2) / (X_scaled.shape[0] * valeurs_propres[:3])
    df_contribution = pd.DataFrame(contribution.round(4), index=df.index, columns=["PC1", "PC2", "PC3"])
    print("\n📌 Contribution des villes à l’inertie des axes :\n", df_contribution)

def afficher_cercle_correlation():
    """ Affiche le cercle des corrélations (PC1 vs PC2 et PC1 vs PC3). """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    df_coord_var = pd.DataFrame(vecteurs_propres[:, :3] * np.sqrt(valeurs_propres[:3]), index=df.columns, columns=["PC1", "PC2", "PC3"])
    
    for i, (ax, (pcx, pcy)) in enumerate(zip(axes, [("PC1", "PC2"), ("PC1", "PC3")])):
        ax.scatter(df_coord_var[pcx], df_coord_var[pcy], c='red')
        for i, var in enumerate(df_coord_var.index):
            ax.text(df_coord_var.iloc[i, 0], df_coord_var.iloc[i, 1], var, fontsize=12)
        ax.axhline(0, color='gray', linestyle='--')
        ax.axvline(0, color='gray', linestyle='--')
        ax.set_xlabel(pcx)
        ax.set_ylabel(pcy)
        ax.set_title(f"Cercle des Corrélations ({pcx} vs {pcy})")

    plt.tight_layout()
    plt.show()

def afficher_biplot():
    """ Affiche le biplot (projection des villes et variables). """
    df_coord_var = pd.DataFrame(vecteurs_propres[:, :3] * np.sqrt(valeurs_propres[:3]), index=df.columns, columns=["PC1", "PC2", "PC3"])

    plt.figure(figsize=(8, 6))
    plt.scatter(X_projected[:, 0], X_projected[:, 1], c='blue', alpha=0.7)
    for i, ville in enumerate(df.index):
        plt.text(X_projected[i, 0], X_projected[i, 1], ville, fontsize=10, color='blue')

    for i, var in enumerate(df_coord_var.index):
        plt.arrow(0, 0, df_coord_var.iloc[i, 0], df_coord_var.iloc[i, 1], color='red', alpha=0.5)
        plt.text(df_coord_var.iloc[i, 0], df_coord_var.iloc[i, 1], var, fontsize=12, color='red')

    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.title("Biplot : Villes et Variables")
    plt.show()

# ----------------------
# 3️⃣ Affichage des résultats
# ----------------------
afficher_distance_carre()
afficher_cos2()
afficher_contribution()
afficher_cercle_correlation()
afficher_biplot()

def afficher_contribution():
    """ Calcule et affiche la contribution des villes à l’inertie des axes, ainsi que la moyenne des contributions. """
    contribution = (X_projected[:, :3] ** 2) / (X_scaled.shape[0] * valeurs_propres[:3])
    df_contribution = pd.DataFrame(contribution.round(4), index=df.index, columns=["PC1", "PC2", "PC3"])
    
    # Calcul de la moyenne des contributions
    moyenne_contribution = df_contribution.mean().round(4)
    
    print("\n📌 Contribution des villes à l’inertie des axes :\n", df_contribution)
    print("\n📌 Moyenne des contributions par axe :\n", moyenne_contribution)

# Exécuter la fonction mise à jour
afficher_contribution()