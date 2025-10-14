#  Traffic Sign Recognition (CS50's AI )

This project is part of Harvard’s **CS50’s Introduction to Artificial Intelligence with Python** course.  
It trains a **Convolutional Neural Network (CNN)** using TensorFlow to classify **traffic sign images** into 43 different categories.

---

##  Overview

The goal of this project is to build and train a deep learning model that can accurately recognize traffic signs from images.  
It uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset or any dataset organized in the same folder structure.

---

##  Dataset Structure

The dataset folder (`data_directory`) should contain 43 subfolders, one for each traffic sign category (0–42).  
Each subfolder contains the images for that sign.

data/

├── 0/

│ ├── 00000_00000.ppm

│ ├── 00000_00001.ppm

│ └── ...

├── 1/

│ ├── 00001_00000.ppm

│ └── ...

...

└── 42/

├── 00042_00000.ppm

└── ...


---

##  How It Works

1. **Load Data:**  
   - Each image is read from the dataset, resized to `30x30`, and normalized (values scaled between 0 and 1).  
   - The folder name determines the label (e.g., folder `5` → label `5`).

2. **Split Data:**  
   - 60% of images are used for training.  
   - 40% are used for testing.

3. **Build CNN Model:**  
   The model architecture includes:
   - Two `Conv2D` layers with 64 filters each (3×3 kernel, ReLU activation)
   - Two `MaxPooling2D` layers (2×2)
   - One fully connected (`Dense`) layer with 512 neurons (ReLU activation)
   - A `Dropout(0.5)` layer to prevent overfitting
   - An output layer with `NUM_CATEGORIES = 43` neurons (softmax activation)

4. **Train the Model:**  
   - Optimizer: `adam`  
   - Loss: `categorical_crossentropy`  
   - Metric: `accuracy`  
   - Trained for 10 epochs.

5. **Evaluate and Save:**  
   - The model’s accuracy is evaluated on the test set.  
   - Optionally saved to a file (e.g., `model.h5`).

---

##  Requirements

Make sure you have the following installed:

```bash
pip install tensorflow opencv-python scikit-learn numpy
