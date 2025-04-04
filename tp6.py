import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

# Initialiser MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Dossier contenant les images
image_folder = r"C:\Users\AYOO INFORMATIQUE\Desktop\AD\visage" 

# Liste pour stocker les features extraites
data = []

# Liste des points d'intérêt et leurs vrais noms
key_points = {
    1: "Milieu_Oeil_Droit",
    33: "Coin_Externe_Oeil_Droit",
    133: "Coin_Externe_Oeil_Gauche",
    362: "Coin_Interne_Oeil_Gauche",
    263: "Coin_Interne_Oeil_Droit",
    61: "Coin_Gauche_Bouche",
    291: "Coin_Droit_Bouche",
    0: "Milieu_Front",
    17: "Pointe_Menton",
    168: "Bout_Nez"
}

# Traitement de chaque image
for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    image = cv2.imread(img_path)
    if image is None:
        continue  # Ignorer si l'image n'est pas valide

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            features = []
            for idx in key_points:
                landmark = face_landmarks.landmark[idx]
                features.extend([landmark.x, landmark.y, landmark.z])

            data.append([img_name] + features)  # Ajouter les features avec le nom de l'image

# Création du DataFrame avec des noms clairs
columns = ["Image"] + [f"{name}_{c}" for name in key_points.values() for c in ["x", "y", "z"]]
df = pd.DataFrame(data, columns=columns)

# Sauvegarde des features
df.to_csv("face_features.csv", index=False)
print("Extraction des features terminée ! Données enregistrées sous 'face_features.csv'")





# # import pandas as pd
# # import numpy as np
# # from sklearn.decomposition import PCA
# # from sklearn.preprocessing import StandardScaler
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # import plotly.express as px

# # # Charger les données
# # df = pd.read_csv("face_features.csv")
# # X = df.drop(columns=["Image"])
# # print(X)
# # # Standardisation
# # scaler = StandardScaler()
# # X_scaled = scaler.fit_transform(X)

# # # ACP
# # pca = PCA(n_components=X.shape[1])
# # X_pca = pca.fit_transform(X_scaled)

# # # Pourcentage de variance expliquée
# # explained_var = pca.explained_variance_ratio_
# # cumulative_var = np.cumsum(explained_var)

# # # Coordonnées des individus
# # individuals_coord = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
# # individuals_coord["Image"] = df["Image"]

# # # Coordonnées des variables
# # components = pd.DataFrame(pca.components_[:2, :].T, columns=['PC1', 'PC2'], index=X.columns)

# # # cos² (qualité de représentation des individus)
# # cos2 = (X_pca[:, :2] ** 2)
# # cos2 = pd.DataFrame(cos2 / np.sum(X_pca ** 2, axis=1, keepdims=True), columns=['cos2_PC1', 'cos2_PC2'])

# # # Contribution des individus à l'inertie des axes
# # total_inertia = np.sum(np.var(X_pca, axis=0))
# # contrib_indiv = pd.DataFrame((X_pca[:, :2] ** 2) / total_inertia * 100, columns=['contrib_PC1', 'contrib_PC2'])

# # # Fusion pour analyse
# # individuals = pd.concat([individuals_coord, cos2, contrib_indiv], axis=1)

# # print("✅ Qualité de représentation (cos²) :")
# # print(cos2.head())

# # print("\n✅ Contribution à l'inertie des axes :")
# # print(contrib_indiv.head())

# # print("\n✅ Coordonnées des variables :")
# # print(components.head())


# # plt.figure(figsize=(8,8))
# # plt.axhline(0, color='grey', lw=1)
# # plt.axvline(0, color='grey', lw=1)
# # circle = plt.Circle((0, 0), 1, color='blue', fill=False)
# # plt.gca().add_artist(circle)

# # for i in range(len(components)):
# #     plt.arrow(0, 0, components.PC1[i], components.PC2[i], 
# #               color='r', alpha=0.5)
# #     plt.text(components.PC1[i]*1.1, components.PC2[i]*1.1, 
# #              components.index[i], color='g', ha='center', va='center')

# # plt.title("Cercle de corrélation (Variables)")
# # plt.xlabel("PC1")
# # plt.ylabel("PC2")
# # plt.grid()
# # plt.axis('equal')
# # plt.show()

# # plt.figure(figsize=(10,6))
# # sns.scatterplot(data=individuals, x="PC1", y="PC2", hue="Image", palette="Set2", legend=False)
# # plt.title("Carte des individus (PC1 vs PC2)")
# # plt.xlabel("PC1")
# # plt.ylabel("PC2")
# # plt.axhline(0, color='grey', lw=1)
# # plt.axvline(0, color='grey', lw=1)
# # plt.grid()
# # plt.show()


# # fig = px.scatter(individuals, x="PC1", y="PC2", text="Image", title="Biplot (Individus + Variables)")
# # # Ajouter les variables
# # for i in range(len(components)):
# #     fig.add_shape(type='line',
# #                   x0=0, y0=0,
# #                   x1=components.PC1[i]*5, y1=components.PC2[i]*5,
# #                   line=dict(color='red'))
# #     fig.add_annotation(x=components.PC1[i]*5,
# #                        y=components.PC2[i]*5,
# #                        ax=0, ay=0,
# #                        xanchor="center", yanchor="bottom",
# #                        text=components.index[i],
# #                        showarrow=False,
# #                        font=dict(color="red"))

# # fig.show()
















# import pandas as pd
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Charger les données
# df = pd.read_csv("face_features.csv")
# X = df.drop(columns=["Image"])  # On enlève la colonne des noms d'images

# # Standardisation (centrer-réduire)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Appliquer l'ACP
# pca = PCA(n_components=2)  # On garde 2 axes principaux pour la visualisation
# X_pca = pca.fit_transform(X_scaled)

# # Créer un DataFrame avec les nouvelles coordonnées des individus
# individuals_coord = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
# individuals_coord["Image"] = df["Image"]

# # Coordonnées des variables (vecteurs propres)
# variables_coord = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=X.columns)

# # Contributions à l'inertie de chaque axe principal
# explained_var = pca.explained_variance_ratio_ * 100  # Pourcentage de variance expliquée

# # Qualité de représentation (cos²) pour chaque individu
# cos2 = (X_pca ** 2) / np.sum(X_pca ** 2, axis=1, keepdims=True)
# cos2 = pd.DataFrame(cos2, columns=["cos2_PC1", "cos2_PC2"])

# # Affichage des résultats
# print("\n✅ Coordonnées des individus :")
# print(individuals_coord.head())

# print("\n✅ Coordonnées des variables :")
# print(variables_coord.head())

# print("\n✅ Contributions à l'inertie :")
# print(pd.DataFrame(explained_var, index=["PC1", "PC2"], columns=["Contribution (%)"]))

# print("\n✅ Qualité de représentation (cos²) :")
# print(cos2.head())





# plt.figure(figsize=(10,6))
# sns.scatterplot(data=individuals_coord, x="PC1", y="PC2", hue="Image", palette="Set2", legend=False)
# plt.axhline(0, color='grey', lw=1)
# plt.axvline(0, color='grey', lw=1)
# plt.title("Projection des individus sur PC1 et PC2")
# plt.xlabel("Axe PC1")
# plt.ylabel("Axe PC2")
# plt.grid()
# plt.show()


# plt.figure(figsize=(8,8))
# plt.axhline(0, color='grey', lw=1)
# plt.axvline(0, color='grey', lw=1)
# circle = plt.Circle((0, 0), 1, color='blue', fill=False)
# plt.gca().add_artist(circle)

# for i in range(len(variables_coord)):
#     plt.arrow(0, 0, variables_coord.PC1[i], variables_coord.PC2[i], 
#               color='r', alpha=0.5)
#     plt.text(variables_coord.PC1[i]*1.1, variables_coord.PC2[i]*1.1, 
#              variables_coord.index[i], color='g', ha='center', va='center')

# plt.title("Cercle de corrélation des variables")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.grid()
# plt.axis('equal')
# plt.show()



# plt.figure(figsize=(8,4))
# plt.bar(["PC1", "PC2"], explained_var, color=["blue", "red"])
# plt.ylabel("Pourcentage de variance expliquée")
# plt.title("Contribution de chaque axe principal")
# plt.show()


# # Calculer les contributions des variables sur les axes principaux (PC1, PC2)
# abs_components = np.abs(pca.components_)  # Valeurs absolues des composantes
# contributions = pd.DataFrame(abs_components, columns=["PC1", "PC2"], index=X.columns)

# # Trouver les 4 variables avec les contributions les plus fortes sur les axes PC1 et PC2
# top_variables = contributions.sum(axis=1).nlargest(4).index
# print("\n✅ Top 4 variables les plus influentes :")
# print(top_variables)


# # Extraire les coordonnées des variables les plus importantes
# top_variables_coord = variables_coord.loc[top_variables]

# # Visualisation avec la qualité de représentation (cos²) sur chaque axe, et les 4 variables importantes
# plt.figure(figsize=(10,6))

# # Projeter les individus avec la qualité de représentation (cos²) comme taille des points
# sns.scatterplot(data=individuals_coord, x="PC1", y="PC2", hue="cos2_PC1", palette="coolwarm", size="cos2_PC1", sizes=(50, 200), legend=False)

# # Ajouter les 4 variables les plus importantes (flèches)
# for i in range(len(top_variables_coord)):
#     plt.arrow(0, 0, top_variables_coord.PC1[i]*5, top_variables_coord.PC2[i]*5, color='red', alpha=0.5)
#     plt.text(top_variables_coord.PC1[i]*5, top_variables_coord.PC2[i]*5, top_variables_coord.index[i], color='black', ha='center', va='center')

# plt.axhline(0, color='grey', lw=1)
# plt.axvline(0, color='grey', lw=1)
# plt.title("Biplot avec les 4 variables les plus influentes et qualité de représentation (cos²)")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.grid()
# plt.show()




# import pandas as pd
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px

# # Charger les données
# df = pd.read_csv("face_features.csv")
# X = df.drop(columns=["Image"])

# # Standardisation des données
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # ACP avec n composants égaux aux variables d'entrée
# pca = PCA(n_components=X.shape[1])
# X_pca = pca.fit_transform(X_scaled)

# # Calcul du pourcentage de variance expliquée par chaque composant principal
# explained_var = pca.explained_variance_ratio_
# cumulative_var = np.cumsum(explained_var)

# # Coordonnées des individus sur les deux premiers axes
# individuals_coord = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
# individuals_coord["Image"] = df["Image"]

# # Coordonnées des variables (composantes principales)
# components = pd.DataFrame(pca.components_[:2, :].T, columns=['PC1', 'PC2'], index=X.columns)

# # Calcul de la qualité de la représentation (cos²) des individus sur les axes
# cos2 = (X_pca[:, :2] ** 2)
# cos2 = pd.DataFrame(cos2 / np.sum(X_pca ** 2, axis=1, keepdims=True), columns=['cos2_PC1', 'cos2_PC2'])

# # Contribution des individus à l'inertie des axes
# total_inertia = np.sum(np.var(X_pca, axis=0))
# contrib_indiv = pd.DataFrame((X_pca[:, :2] ** 2) / total_inertia * 100, columns=['contrib_PC1', 'contrib_PC2'])

# # Fusion des résultats pour analyse complète
# individuals = pd.concat([individuals_coord, cos2, contrib_indiv], axis=1)

# # Affichage des résultats
# print("✅ Qualité de représentation (cos²) :")
# print(cos2.head())

# print("\n✅ Contribution à l'inertie des axes :")
# print(contrib_indiv.head())

# print("\n✅ Coordonnées des variables :")
# print(components.head())

# # Visualisation des résultats
# import numpy as np
# import matplotlib.pyplot as plt

# # Création de la figure
# fig, ax = plt.subplots(figsize=(8, 8))

# # Tracer le cercle unitaire (cercle de corrélation)
# circle = plt.Circle((0, 0), 0.39, color='blue', fill=False, linestyle='solid', linewidth=0.5)
# ax.add_patch(circle)  # Ajouter le cercle au graphique

# # Dessiner les axes
# plt.axhline(0, color='black', lw=1.2)
# plt.axvline(0, color='black', lw=1.2)

# # Dessiner les flèches des variables
# for i in range(len(components)):
#     x = components.PC1[i]  # Coordonnée sur PC1
#     y = components.PC2[i]  # Coordonnée sur PC2
    
#     # Dessiner la flèche pour chaque variable
#     plt.arrow(0, 0, x, y, head_width=0.025, head_length=0.025, color='red', alpha=0.75, linewidth=0.5)
    
#     # Afficher le nom de la variable
#     plt.text(x * 1.15, y * 1.15, components.index[i], color='green', ha='center', va='center', fontsize=6, fontweight='bold')

# # Paramètres du graphique
# plt.xlabel(f"PC1 ({round(explained_var[0] * 100, 2)}%)", fontsize=12, fontweight='bold')
# plt.ylabel(f"PC2 ({round(explained_var[1] * 100, 2)}%)", fontsize=12, fontweight='bold')
# plt.title("Cercle de Corrélation des Variables", fontsize=14, fontweight='bold')

# # Ajuster les limites du graphique pour bien voir le cercle
# plt.xlim(-1.2, 1.2)
# plt.ylim(-1.2, 1.2)

# # Activer le quadrillage et garder une échelle égale
# plt.grid(linestyle='dashed', alpha=0.6)
# plt.axis('equal')  # Assurer que le cercle reste bien rond
# plt.show()


# # Carte des individus (PC1 vs PC2)
# plt.figure(figsize=(10,6))
# sns.scatterplot(data=individuals, x="PC1", y="PC2", hue="Image", palette="Set2", legend=False)
# plt.title("Carte des individus (PC1 vs PC2)")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.axhline(0, color='grey', lw=1)
# plt.axvline(0, color='grey', lw=1)
# plt.grid()
# plt.show()

# # # Biplot interactif avec Plotly
# # fig = px.scatter(individuals, x="PC1", y="PC2", text="Image", title="Biplot (Individus + Variables)")
# # for i in range(len(components)):
# #     fig.add_shape(type='line',
# #                   x0=0, y0=0,
# #                   x1=components.PC1[i]*5, y1=components.PC2[i]*5,
# #                   line=dict(color='red'))
# #     fig.add_annotation(x=components.PC1[i]*5,
# #                        y=components.PC2[i]*5,
# #                        ax=0, ay=0,
# #                        xanchor="center", yanchor="bottom",
# #                        text=components.index[i],
# #                        showarrow=False,
# #                        font=dict(color="red"))

# # fig.show()
