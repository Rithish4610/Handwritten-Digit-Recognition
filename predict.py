# PREDICT FROM YOUR OWN IMAGE
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

def predict_uploaded_image(image_path):
    """
    Predict digit from an uploaded image file.
    """
    # Load the saved model
    print("Loading model...")
    loaded_model = tf.keras.models.load_model('my_simple_model.h5')
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: File '{image_path}' not found!")
        print(f"\nPlease save your handwritten digit image in:")
        print(f"   {os.getcwd()}")
        return None
    
    print(f"Processing image: {image_path}")
    
    # Load and preprocess the image
    img = Image.open(image_path)
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Resize to 28x28 pixels (MNIST format)
    img = img.resize((28, 28))
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # MNIST has white digit on black background
    # If your image has black digit on white background, invert it
    if np.mean(img_array) > 127:  # Light background detected
        img_array = 255 - img_array  # Invert colors
    
    # Normalize to 0-1 range
    img_array = img_array / 255.0
    
    # Reshape for model input (add batch dimension)
    img_array = img_array.reshape(1, 28, 28)
    
    # Make prediction
    prediction = loaded_model.predict(img_array, verbose=0)
    predicted_digit = np.argmax(prediction[0])
    confidence = prediction[0][predicted_digit] * 100
    
    # Display the processed image and prediction
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(Image.open(image_path), cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_array.reshape(28, 28), cmap='gray')
    plt.title(f"Processed (28x28)")
    plt.axis('off')
    
    plt.suptitle(f"🎯 Predicted Digit: {predicted_digit} (Confidence: {confidence:.1f}%)", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print(f"\n🔢 Prediction Results:")
    print(f"   Predicted Digit: {predicted_digit}")
    print(f"   Confidence: {confidence:.1f}%")
    print(f"\n   All probabilities:")
    for i, prob in enumerate(prediction[0]):
        bar = "█" * int(prob * 20)
        print(f"   {i}: {bar} {prob*100:.1f}%")
    
    return predicted_digit

# Run prediction if image path provided as argument
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        predict_uploaded_image(image_path)
    else:
        print("="*50)
        print("📷 HANDWRITTEN DIGIT PREDICTOR")
        print("="*50)
        print("\nUsage: py -3.11 predict.py <image_path>")
        print("\nExample:")
        print("   py -3.11 predict.py my_digit.jpg")
        print("   py -3.11 predict.py test.png")
        print("\nSupported formats: .png, .jpg, .jpeg, .bmp")
        print("="*50)
