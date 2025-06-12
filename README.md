# Microsense

**Microsense** is a lightweight and efficient AI-powered image classification system designed to identify various types of microorganisms from microscopic images. Built with modularity and ease of use in mind, the platform helps students, researchers, and lab technicians quickly categorize and visualize microorganism samples using machine learning techniques.

## 🧪 Features

  - Dataset preprocessing and transformation to `.npz` format for optimized model training.
  - Interactive image viewer to explore dataset visually.
  - Ready-to-train architecture using deep learning for classification.
  - Well-structured project suitable for education and research.

## 📁 Project Structure
```bash
  micro_organism_project/
  ├── dataset/
  │ └── bacteria_dataset/ # Raw image data (.jpg)
  ├── program/
  │ ├── preprocess.py # Converts images to .npz format
  │ └── view_dataset.py # Visual viewer for dataset browsing
```

## ⚙️ Requirements

  - Python 3.8+
  - NumPy
  - OpenCV (`cv2`)
  - Matplotlib

Install dependencies:

  ```bash
  pip install numpy opencv-python matplotlib
  ```
## 🚀 Getting Started

  1. Clone the Repository
     ```bash
     git clone https://github.com/yourusername/Microsense.git
     cd Microsense
     ```
  2. Preprocess the Dataset

      Converts the raw .jpg files into .npz format for efficient access.
```bash
       cd program
       python preprocess.py
```
  3. View the Dataset

      Use the built-in dataset viewer to interactively inspect the image samples.
     ```bash
         python view_dataset.py
     ```
## 🎯 Goals

  Improve microorganism identification using deep learning.
  Build a clear dataset pipeline for medical/biotech image analysis.
  Make AI tools more accessible in microbiology education.

## 📝 License
  
  This project is licensed under the MIT License. See the LICENSE file for more details.
     
     
