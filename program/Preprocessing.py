import os
import cv2
import numpy as np
import subprocess

# Dataset path
image_dir = r"C:\Users\rafiq\Downloads\Micro Organism project\Dataset"
img_size = (224, 224)

data = []
labels = []

# Walk through subdirectories
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(root, file)

            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Couldn't read image {img_path}")
                continue

            image = cv2.resize(image, img_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            data.append(image)
            labels.append(file)  # Use filename as label

# Convert to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Save the .npz dataset
os.makedirs("program", exist_ok=True)
save_path = os.path.join("program", "micro_dataset.npz")
np.savez(save_path, images=data, labels=labels)
print(f"Dataset saved as {save_path}")

# Automatically run view.py
print("Launching dataset viewer...")
subprocess.run(["python", os.path.join("program", "view.py")])
