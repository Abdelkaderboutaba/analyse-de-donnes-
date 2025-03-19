import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq

# Étape 1 : Charger les images depuis un dossier
def load_images_from_folder(folder, target_size=(128, 128)):
    """
    Charge les images depuis un dossier, les convertit en niveaux de gris et les redimensionne.
    :param folder: Chemin du dossier contenant les images.
    :param target_size: Taille cible des images (par défaut 128x128).
    :return: Liste des images prétraitées.
    """
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Charger en niveaux de gris
        if img is not None:
            img = cv2.resize(img, target_size)  # Redimensionner l'image
            images.append(img)
    return np.array(images)

# Étape 2 : Extraire des descripteurs SIFT
def extract_sift_features(images):
    """
    Extrait des descripteurs SIFT pour chaque image.
    :param images: Liste des images en niveaux de gris.
    :return: Liste des descripteurs SIFT pour chaque image.
    """
    sift = cv2.SIFT_create()  # Créer un objet SIFT
    descriptors_list = []
    for img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)  # Détecter les keypoints et calculer les descripteurs
        if descriptors is not None:
            descriptors_list.append(descriptors)
    return descriptors_list

# Étape 3 : Créer un Bag of Visual Words (BoVW)
def create_bovw(descriptors_list, n_clusters=100):
    """
    Crée un Bag of Visual Words (BoVW) à partir des descripteurs SIFT.
    :param descriptors_list: Liste des descripteurs SIFT pour chaque image.
    :param n_clusters: Nombre de clusters (mots visuels) pour le BoVW.
    :return: Tableau NumPy des caractéristiques BoVW.
    """
    # Concaténer tous les descripteurs
    all_descriptors = np.vstack(descriptors_list)

    # Appliquer k-means pour créer un vocabulaire visuel
    codebook, _ = kmeans(all_descriptors, n_clusters)

    # Créer un histogramme pour chaque image
    features = []
    for descriptors in descriptors_list:
        if descriptors is not None:
            words, _ = vq(descriptors, codebook)  # Associer chaque descripteur au mot visuel le plus proche
            hist, _ = np.histogram(words, bins=np.arange(n_clusters + 1))  # Calculer l'histogramme
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)  # Normalisation de l'histogramme
            features.append(hist)
    return np.array(features)

# Étape 4 : Appliquer l'ACPN
def apply_acpn(features, n_components=2):
    """
    Applique l'ACPN (PCA avec standardisation) sur les caractéristiques.
    :param features: Tableau NumPy des caractéristiques.
    :param n_components: Nombre de composantes principales à conserver (par défaut 2).
    :return: Tableau NumPy des caractéristiques réduites, variance expliquée, et composantes principales.
    """
    # Standardisation des caractéristiques
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(features)

    # Application de l'ACPN
    acpn = PCA(n_components=n_components)
    features_reduced = acpn.fit_transform(features_standardized)
    return features_reduced, acpn.explained_variance_ratio_, acpn.components_

# Étape 5 : Afficher le cercle de corrélation
def plot_correlation_circle(components, feature_names):
    """
    Affiche le cercle de corrélation pour les deux premières composantes principales.
    :param components: Composantes principales (axes de l'ACPN).
    :param feature_names: Noms des caractéristiques (mots visuels).
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, (x, y) in enumerate(components[:2].T):  # Prendre les deux premières composantes
        ax.arrow(0, 0, x, y, head_width=0.05, head_length=0.05, fc='r', ec='r')
        ax.text(x, y, feature_names[i], color='b', fontsize=12)
    circle = plt.Circle((0, 0), 1, color='blue', fill=False)
    ax.add_artist(circle)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Composante Principale 1")
    ax.set_ylabel("Composante Principale 2")
    ax.set_title("Cercle de Corrélation")
    plt.show()

# Étape 6 : Afficher les résultats de l'ACPN
def plot_acpn_results(features_reduced, labels=None):
    """
    Affiche les résultats de l'ACPN dans un graphique 2D.
    :param features_reduced: Tableau NumPy des caractéristiques réduites.
    :param labels: Labels des classes (optionnel).
    """
    plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c=labels)
    plt.title("Résultats de l'ACPN")
    plt.xlabel("Composante Principale 1")
    plt.ylabel("Composante Principale 2")
    plt.show()

# Fonction principale
def main():
    # Chemin du dossier contenant les images
    folder_path = "faces"  # Remplacez par le chemin de votre dossier

    # Charger les images
    images = load_images_from_folder(folder_path)
    print(f"{len(images)} images chargées.")

    # Extraire les descripteurs SIFT
    descriptors_list = extract_sift_features(images)
    print(f"Descripteurs SIFT extraits pour {len(descriptors_list)} images.")

    # Créer un Bag of Visual Words (BoVW)
    n_clusters = 100  # Nombre de mots visuels
    features = create_bovw(descriptors_list, n_clusters)
    print(f"Caractéristiques BoVW extraites : {features.shape}")

    # Appliquer l'ACPN
    features_reduced, variance_explained, components = apply_acpn(features)
    print(f"Variance expliquée par chaque composante : {variance_explained}")

    # Afficher le cercle de corrélation
    feature_names = [f"Word {i}" for i in range(n_clusters)]  # Noms des mots visuels
    plot_correlation_circle(components, feature_names)

    # Afficher les résultats de l'ACPN
    plot_acpn_results(features_reduced)

# Exécuter le programme
if __name__ == "__main__":
    main()