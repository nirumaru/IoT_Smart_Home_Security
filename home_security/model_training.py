import os
import pickle
import cv2
import face_recognition
from imutils import paths

# --- CONFIGURATION ---
DATASET_PATH = "dataset"  # Folder containing your images
ENCODINGS_FILE = "encodings.pickle"  # Output file name
DETECTION_METHOD = "hog"  # Use "hog" for Pi (fast), "cnn" for GPU (accurate)

print("[INFO] Start processing faces...")

# 1. Grab the paths to the input images in our dataset
imagePaths = list(paths.list_images(DATASET_PATH))
knownEncodings = []
knownNames = []

# 2. Loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # Extract the person name from the folder name
    # Expected structure: dataset/name/image.jpg
    name = imagePath.split(os.path.sep)[-2]

    print(f"[INFO] Processing image {i + 1}/{len(imagePaths)}: {name}")

    # Load the input image and convert it from BGR (OpenCV) to RGB (dlib)
    image = cv2.imread(imagePath)
    if image is None:
        print(f"[WARNING] Could not read image: {imagePath}. Skipping.")
        continue

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect the (x, y)-coordinates of the bounding boxes corresponding
    # to each face in the input image
    boxes = face_recognition.face_locations(rgb, model=DETECTION_METHOD)

    # Compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)

    # Loop over the encodings (in case multiple faces are in one image)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

# 3. Dump the facial encodings + names to disk
print("[INFO] Serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}

try:
    with open(ENCODINGS_FILE, "wb") as f:
        f.write(pickle.dumps(data))
    print(f"[INFO] Success! Encodings saved to '{ENCODINGS_FILE}'")
except Exception as e:
    print(f"[ERROR] Could not write pickle file: {e}")