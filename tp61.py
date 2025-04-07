import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Folder containing face images
image_folder = r"C:\Users\AYOO INFORMATIQUE\Desktop\AD\visage"

# Improved key landmarks for emotion-related features
# Includes eyes, mouth, nose tip, eyebrows
key_points = {
    33: "Right_Eye_Outer",
    263: "Left_Eye_Outer",
    133: "Right_Eye_Inner",
    362: "Left_Eye_Inner",
    159: "Right_Eye_Upper_Lid",
    145: "Right_Eye_Lower_Lid",
    386: "Left_Eye_Upper_Lid",
    374: "Left_Eye_Lower_Lid",
    61: "Mouth_Left",
    291: "Mouth_Right",
    13: "Mouth_Upper",
    14: "Mouth_Lower",
    0: "Forehead_Center",
    17: "Chin_Tip",
    168: "Nose_Tip",
    70: "Right_Eyebrow",
    300: "Left_Eyebrow"
}

# Storage for features
data = []

# Process each image
for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    image = cv2.imread(img_path)
    if image is None:
        continue

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Extract selected landmarks
            face_data = {}
            for idx, name in key_points.items():
                lm = landmarks[idx]
                face_data[name] = (lm.x, lm.y, lm.z)

            # Normalize coordinates by inter-ocular distance
            eye_dist = np.linalg.norm(
                np.array(face_data["Right_Eye_Outer"][:2]) -
                np.array(face_data["Left_Eye_Outer"][:2])
            )
            if eye_dist == 0:
                continue  # avoid divide-by-zero

            # Build feature list: normalized x, y, z
            features = []
            for name in key_points.values():
                x, y, z = face_data[name]
                features.extend([x / eye_dist, y / eye_dist, z / eye_dist])

            data.append([img_name] + features)

# Build column names
columns = ["Image"] + [f"{name}_{c}" for name in key_points.values() for c in ["x", "y", "z"]]

# Save to CSV
df = pd.DataFrame(data, columns=columns)
df.to_csv("face_features.csv", index=False)
print("✅ Features extracted and saved to 'improved_face_features.csv'")


