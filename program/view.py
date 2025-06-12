import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from sklearn.metrics.pairwise import cosine_similarity
from tkinter import Tk, filedialog
import cv2
import os

# --- Load Dataset ---
npz_path = r"program/micro_dataset.npz"
data = np.load(npz_path, allow_pickle=True)
images_all = data["images"]
labels_all = data["labels"]  # Should be image filenames

# --- Image Feature Extraction Parameters ---
img_size = (224, 224)

def extract_features(image):
    image_resized = cv2.resize(image, img_size)
    hist = cv2.calcHist([image_resized], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# --- Precompute Features ---
dataset_features = np.array([extract_features(img) for img in images_all])

# --- Initial Setup ---
filtered_images = list(images_all)
filtered_labels = list(labels_all)
index = {"value": 0}

# --- Plot Setup ---
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.3)
img_display = ax.imshow(filtered_images[0])
ax.axis("off")

# --- Label Formatter ---
def get_label_text(label):
    try:
        return os.path.splitext(os.path.basename(str(label)))[0]
    except:
        return "Unknown"

# --- Set Initial Title ---
label_text = get_label_text(filtered_labels[0])
title = ax.set_title(f"Image 1 of {len(filtered_images)} | Label: {label_text}")

# --- UI Elements ---
ax_prev = plt.axes([0.2, 0.17, 0.15, 0.07])
btn_prev = Button(ax_prev, 'Previous')

ax_next = plt.axes([0.65, 0.17, 0.15, 0.07])
btn_next = Button(ax_next, 'Next')

ax_box = plt.axes([0.1, 0.08, 0.4, 0.06])
text_box = TextBox(ax_box, "Label :", initial="")

ax_search = plt.axes([0.52, 0.08, 0.15, 0.06])
btn_search = Button(ax_search, "Search")

ax_compare = plt.axes([0.7, 0.08, 0.15, 0.06])
btn_compare = Button(ax_compare, "Compare")

# --- Update Display ---
def update_display():
    total = len(filtered_images)
    if total == 0:
        img_display.set_data(np.zeros_like(images_all[0]))
        title.set_text("No images found")
    else:
        current_image = filtered_images[index["value"]]
        img_display.set_data(current_image)
        label_text = get_label_text(filtered_labels[index["value"]])
        title.set_text(f"Image {index['value'] + 1} of {total} | Label: {label_text}")
    fig.canvas.draw_idle()

# --- Navigation ---
def next_image(event):
    if index["value"] < len(filtered_images) - 1:
        index["value"] += 1
        update_display()

def prev_image(event):
    if index["value"] > 0:
        index["value"] -= 1
        update_display()

# --- Search by Label ---
def filter_images(event):
    global filtered_images, filtered_labels
    label_input = text_box.text.strip().lower()
    if label_input == "":
        filtered_images = list(images_all)
        filtered_labels = list(labels_all)
    else:
        mask = [label_input in str(lbl).lower() for lbl in labels_all]
        filtered_images = [img for img, m in zip(images_all, mask) if m]
        filtered_labels = [lbl for lbl, m in zip(labels_all, mask) if m]

    index["value"] = 0
    update_display()

# --- Compare with Uploaded Image ---
def compare_image(event):
    Tk().withdraw()
    file_path = filedialog.askopenfilename(title="Select an image to compare")
    if not file_path:
        print("No file selected.")
        return

    input_img = cv2.imread(file_path)
    if input_img is None:
        print("Failed to load image.")
        return

    input_img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_feat = extract_features(input_img_rgb)

    sims = cosine_similarity([input_feat], dataset_features)[0]
    best_idx = np.argmax(sims)

    # --- Show Side-by-Side Comparison ---
    fig2, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(input_img_rgb)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    best_label = get_label_text(labels_all[best_idx])
    axes[1].imshow(images_all[best_idx])
    axes[1].set_title(f"Matched Image\nLabel: {best_label}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

# --- Connect Buttons ---
btn_next.on_clicked(next_image)
btn_prev.on_clicked(prev_image)
btn_search.on_clicked(filter_images)
btn_compare.on_clicked(compare_image)

plt.show()
