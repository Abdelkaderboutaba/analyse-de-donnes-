import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh  # Load Face Mesh module
face_mesh = mp_face_mesh.FaceMesh()  # Create Face Mesh detector

image = cv2.imread("faces\Amitabh Bachchan_44.jpg")  # Load the image
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

results = face_mesh.process(rgb_image)  # Detect face landmarks

# Check if face landmarks were detected
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * image.shape[1])  # Convert normalized x to pixel value
            y = int(landmark.y * image.shape[0])  # Convert normalized y to pixel value
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # Draw small circles

# Show the result
cv2.imshow("Face Mesh", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


