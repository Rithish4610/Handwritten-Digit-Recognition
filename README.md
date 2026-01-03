# 🔢 Handwritten Digit Recognition

A Machine Learning project that recognizes handwritten digits (0-9) using Neural Networks and the MNIST dataset.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-97.77%25-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Tips for Best Results](#tips-for-best-results)

---

## 🎯 Overview

This project uses a **Neural Network** to classify images of handwritten digits into their respective numbers (0-9). It is trained on the famous **MNIST dataset** containing 70,000 grayscale images of handwritten digits.

---

## ✨ Features

- ✅ Train a digit recognition model from scratch
- ✅ **Upload your own handwritten digit images** for prediction
- ✅ Automatic image preprocessing (resize, grayscale, normalize)
- ✅ Auto-invert colors (works with both light and dark backgrounds)
- ✅ Display prediction confidence with probability breakdown
- ✅ Save and load trained models

---

## 📁 Project Structure

```
HAND WRITTEN DIGIT RECOGNITION/
│
├── simple_digit_recognizer.py   # Main training script
├── predict.py                   # Prediction script for custom images
├── digit_recognition.ipynb      # Jupyter notebook (CNN version)
├── my_simple_model.h5           # Saved trained model
└── README.md                    # This file
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.11 or compatible version
- pip (Python package manager)

### Install Dependencies

```bash
pip install tensorflow pillow matplotlib numpy
```

---

## 🚀 Usage

### 1. Train the Model

```bash
py -3.11 simple_digit_recognizer.py
```

This will:
- Load the MNIST dataset (60,000 training + 10,000 test images)
- Train the neural network for 5 epochs
- Display test accuracy
- Save the model as `my_simple_model.h5`

### 2. Predict Your Own Handwritten Digit

```bash
py -3.11 predict.py <path_to_your_image>
```

**Example:**
```bash
py -3.11 predict.py "C:\Users\rithi\Downloads\my_digit.jpg"
```

**Output:**
```
🔢 Prediction Results:
   Predicted Digit: 3
   Confidence: 90.1%

   All probabilities:
   0:  0.0%
   1:  0.0%
   2: █ 1.0%
   3: ██████████████████ 90.1%
   ...
```

---

## 🧠 How It Works

### Step-by-Step Process:

| Step | Description |
|------|-------------|
| 1️⃣ **Load Data** | Load 60,000 training images from MNIST dataset |
| 2️⃣ **Preprocess** | Normalize pixel values from 0-255 to 0-1 |
| 3️⃣ **Build Model** | Create neural network with input, hidden, and output layers |
| 4️⃣ **Train** | Feed images through network, adjust weights to minimize errors |
| 5️⃣ **Evaluate** | Test on 10,000 unseen images to measure accuracy |
| 6️⃣ **Predict** | Process new images and output predicted digit |

### Image Processing Pipeline:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Upload    │     │  Grayscale  │     │   Resize    │     │  Normalize  │
│   Image     │ ──► │  Convert    │ ──► │   28×28     │ ──► │   0 to 1    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
                                                                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Predicted  │     │   Softmax   │     │   Neural    │     │   Invert    │
│   Digit     │ ◄── │   Output    │ ◄── │   Network   │ ◄── │  (if needed)│
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

---

## 🏗️ Model Architecture

### Simple Dense Network (simple_digit_recognizer.py)

```
Layer (type)                Output Shape              Param #
═══════════════════════════════════════════════════════════════
Flatten                     (None, 784)               0
Dense (ReLU)                (None, 128)               100,480
Dense (Softmax)             (None, 10)                1,290
═══════════════════════════════════════════════════════════════
Total params: 101,770
```

| Layer | Description |
|-------|-------------|
| **Flatten** | Converts 28×28 image to 784-element vector |
| **Dense (128)** | Hidden layer with 128 neurons, ReLU activation |
| **Dense (10)** | Output layer with 10 neurons (one per digit) |

### CNN Version (digit_recognition.ipynb)

```
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(64) → Dense(10)
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Training Accuracy** | 98.55% |
| **Validation Accuracy** | 97.72% |
| **Test Accuracy** | **97.77%** |
| **Training Time** | ~30 seconds (5 epochs) |

### Training Progress:

| Epoch | Training Acc | Validation Acc |
|-------|-------------|----------------|
| 1 | 92.24% | 96.35% |
| 2 | 96.49% | 97.25% |
| 3 | 97.49% | 97.67% |
| 4 | 98.10% | 97.37% |
| 5 | 98.55% | 97.72% |

---

## 💡 Tips for Best Results

For accurate predictions on your own images:

| Tip | Description |
|-----|-------------|
| ✍️ **Write Clearly** | Use a dark pen/marker on white paper |
| 📷 **Good Lighting** | Ensure even lighting, avoid shadows |
| 🎯 **Center the Digit** | Place digit in the center of the image |
| 📐 **Square Crop** | Crop image to be roughly square |
| 🔲 **High Contrast** | Dark digit on light background (or vice versa) |
| 📏 **Fill the Frame** | Digit should take up most of the image |

**Supported Image Formats:** `.png`, `.jpg`, `.jpeg`, `.bmp`

---

## 🔧 Technologies Used

- **Python 3.11** - Programming language
- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computations
- **Matplotlib** - Visualization
- **Pillow (PIL)** - Image processing
- **MNIST Dataset** - Training data (70,000 handwritten digit images)

---

## 📜 License

This project is open source and available for educational purposes.

---

## 🙋 Author

Created as a beginner-friendly Machine Learning project to demonstrate image classification with neural networks.

---

**⭐ If this project helped you learn, give it a star!**
