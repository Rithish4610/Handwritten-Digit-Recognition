# 🔢 Handwritten Digit Recognition

A Machine Learning project that recognizes handwritten digits (0-9) using Neural Networks and the MNIST dataset.
-

## 🎯 Overview

This project uses a **Neural Network** to classify images of handwritten digits into their respective numbers (0-9). It is trained on the famous **MNIST dataset** containing 70,000 grayscale images of handwritten digits.

## ✨ Features

- ✅ Train a digit recognition model from scratch
- ✅ **Upload your own handwritten digit images** for prediction
- ✅ Automatic image preprocessing (resize, grayscale, normalize)
- ✅ Auto-invert colors (works with both light and dark backgrounds)
- ✅ Display prediction confidence with probability breakdown
- ✅ Save and load trained models

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


| Tip | Description |
|-----|-------------|
| ✍️ **Write Clearly** | Use a dark pen/marker on white paper |
| 📷 **Good Lighting** | Ensure even lighting, avoid shadows |
| 🎯 **Center the Digit** | Place digit in the center of the image |
| 📐 **Square Crop** | Crop image to be roughly square |
| 🔲 **High Contrast** | Dark digit on light background (or vice versa) |
| 📏 **Fill the Frame** | Digit should take up most of the image |

**Supported Image Formats:** `.png`, `.jpg`, `.jpeg`, `.bmp`

## 🔧 Technologies Used

- **Python 3.11** - Programming language
- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computations
- **Matplotlib** - Visualization
- **Pillow (PIL)** - Image processing
- **MNIST Dataset** - Training data (70,000 handwritten digit images)

