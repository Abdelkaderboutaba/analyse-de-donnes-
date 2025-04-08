# import cv2
# import mediapipe as mp
# import numpy as np
# import os
# import pandas as pd

# # Initialize MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# # Folder containing face images
# image_folder = r"C:\Users\AYOO INFORMATIQUE\Desktop\AD\visage"

# # Improved key landmarks for emotion-related features
# # Includes eyes, mouth, nose tip, eyebrows
# key_points = {
#     33: "Right_Eye_Outer",
#     263: "Left_Eye_Outer",
#     133: "Right_Eye_Inner",
#     362: "Left_Eye_Inner",
#     159: "Right_Eye_Upper_Lid",
#     145: "Right_Eye_Lower_Lid",
#     386: "Left_Eye_Upper_Lid",
#     374: "Left_Eye_Lower_Lid",
#     61: "Mouth_Left",
#     291: "Mouth_Right",
#     13: "Mouth_Upper",
#     14: "Mouth_Lower",
#     0: "Forehead_Center",
#     17: "Chin_Tip",
#     168: "Nose_Tip",
#     70: "Right_Eyebrow",
#     300: "Left_Eyebrow"
# }


# # Storage for features
# data = []

# # Process each image
# for img_name in os.listdir(image_folder):
#     img_path = os.path.join(image_folder, img_name)
#     image = cv2.imread(img_path)
#     if image is None:
#         continue

#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_image)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             landmarks = face_landmarks.landmark

#             # Extract selected landmarks
#             face_data = {}
#             for idx, name in key_points.items():
#                 lm = landmarks[idx]
#                 face_data[name] = (lm.x, lm.y, lm.z)

#             # Normalize coordinates by inter-ocular distance
#             eye_dist = np.linalg.norm(
#                 np.array(face_data["Right_Eye_Outer"][:2]) -
#                 np.array(face_data["Left_Eye_Outer"][:2])
#             )
#             if eye_dist == 0:
#                 continue  # avoid divide-by-zero

#             # Build feature list: normalized x, y, z
#             features = []
#             for name in key_points.values():
#                 x, y, z = face_data[name]
#                 features.extend([x / eye_dist, y / eye_dist, z / eye_dist])

#             data.append([img_name] + features)

# # Build column names
# columns = ["Image"] + [f"{name}_{c}" for name in key_points.values() for c in ["x", "y", "z"]]

# # Save to CSV
# df = pd.DataFrame(data, columns=columns)
# df.to_csv("face_features.csv", index=False)
# print("✅ Features extracted and saved to 'improved_face_features.csv'")


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

# === Chargement des données ===
df = pd.read_csv("face_features.csv")

has_expression = "Expression" in df.columns
if has_expression:
    expressions = df["Expression"]
    X = df.drop(columns=["Image", "Expression"])
else:
    X = df.drop(columns=["Image"])

images_names = df["Image"]

# === Standardisation ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === ACP ===
pca = PCA(n_components=X.shape[1])
X_pca = pca.fit_transform(X_scaled)

# === Variance expliquée ===
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)





# ✅ Coordonnées des individus (images)
individuals_coord = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
individuals_coord["Image"] = df["Image"]

# ✅ cos² des individus
cos2 = (X_pca[:, :2] ** 2)
cos2 = pd.DataFrame(cos2 / np.sum(X_pca ** 2, axis=1, keepdims=True), columns=['cos2_PC1', 'cos2_PC2'])

# ✅ Contribution des individus
total_inertia = np.sum(np.var(X_pca, axis=0))
contrib_indiv = pd.DataFrame((X_pca[:, :2] ** 2) / total_inertia * 100, columns=['contrib_PC1', 'contrib_PC2'])

# ✅ Coordonnées des variables (composantes principales)
components = pd.DataFrame(pca.components_[:2, :].T, columns=['PC1', 'PC2'], index=X.columns)

# ✅ cos² des variables
var_cos2 = components ** 2
var_cos2.columns = ['cos2_PC1', 'cos2_PC2']


# ✅ Contribution des variables
eigenvalues = pca.explained_variance_[:2]
contrib_var = components.copy()
contrib_var['contrib_PC1'] = (components['PC1']**2 / eigenvalues[0]) * 100
contrib_var['contrib_PC2'] = (components['PC2']**2 / eigenvalues[1]) * 100

# ✅ Données finales des individus
individuals = pd.concat([individuals_coord, cos2, contrib_indiv], axis=1)


# ✅ Affichage des infos
print("✅ Qualité de représentation (cos²) des individus :")
print(cos2.head())

print("\n✅ Contribution des individus :")
print(contrib_indiv.head())

print("\n✅ Coordonnées des variables :")
print(components.head())

print("\n✅ Qualité de représentation (cos²) des variables :")
print(var_cos2.head())

print("\n✅ Contribution des variables :")
print(contrib_var[['contrib_PC1', 'contrib_PC2']].head())


# === KMeans Clustering ===
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(individuals_coord[["PC1", "PC2"]])
individuals_coord["Cluster"] = clusters

# === Cercle de corrélation ===
components = pd.DataFrame(pca.components_[:2, :].T, columns=['PC1', 'PC2'], index=X.columns)

fig, ax = plt.subplots(figsize=(8, 8))
circle = plt.Circle((0, 0), 0.39, color='blue', fill=False, linestyle='solid', linewidth=0.5)
ax.add_patch(circle)
plt.axhline(0, color='black', lw=1.2)
plt.axvline(0, color='black', lw=1.2)

for i in range(len(components)):
    x, y = components.PC1[i], components.PC2[i]
    plt.arrow(0, 0, x, y, head_width=0.025, head_length=0.025, color='red', alpha=0.75, linewidth=0.5)
    plt.text(x * 1.15, y * 1.15, components.index[i], color='green', ha='center', va='center', fontsize=6, fontweight='bold')

plt.xlabel(f"PC1 ({round(explained_var[0] * 100, 2)}%)", fontsize=12, fontweight='bold')
plt.ylabel(f"PC2 ({round(explained_var[1] * 100, 2)}%)", fontsize=12, fontweight='bold')
plt.title("Cercle de Corrélation des Variables", fontsize=14, fontweight='bold')
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.grid(linestyle='dashed', alpha=0.6)
plt.axis('equal')
plt.show()

# === Affichage des clusters avec les images ===
image_folder = r"C:\\Users\\AYOO INFORMATIQUE\\Desktop\\AD\\visage"

plt.figure(figsize=(12, 8))
ax = plt.gca()

for i, row in individuals_coord.iterrows():
    x, y = row["PC1"], row["PC2"]
    img_path = os.path.join(image_folder, row["Image"])
    if os.path.exists(img_path):
        try:
            img = Image.open(img_path).resize((25, 25))
            ax.imshow(np.array(img), extent=(x-0.3, x+0.3, y-0.3, y+0.3), zorder=1)
        except:
            plt.text(x, y, "?", fontsize=8, ha="center", va="center")
    else:
        plt.text(x, y, "?", fontsize=8, ha="center", va="center")

sns.scatterplot(data=individuals_coord, x="PC1", y="PC2", hue="Cluster", palette="Set1", alpha=0.3, s=80, legend="full", ax=ax)

plt.title("Clusters de visages sur les composantes principales (PCA)", fontsize=14)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.legend(title="Cluster")
plt.show()

# === Comparaison avec les expressions faciales (si présentes) ===
if has_expression:
    plt.figure(figsize=(10,6))
    sns.countplot(data=individuals_coord, x="Cluster", hue="Expression", palette="Set2")
    plt.title("Distribution des expressions faciales par cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Nombre d’images")
    plt.grid(axis='y')
    plt.show()

# === Méthode du coude ===
inertias = []
K_range = range(1, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(individuals_coord[["PC1", "PC2"]])
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Inertie intra-cluster")
plt.title("Méthode du coude (elbow method)")
plt.grid()
plt.show()