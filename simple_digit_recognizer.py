# SIMPLE DIGIT RECOGNIZER - BEGINNER VERSION
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("Setting up... Please wait!")

# 1. LOAD DATA (Simple!)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Loaded {len(x_train)} training images")
print(f"Loaded {len(x_test)} test images")

# 2. SEE SOME IMAGES
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# 3. SIMPLE PREPROCESSING
x_train = x_train / 255.0
x_test = x_test / 255.0

# 4. BUILD SIMPLE MODEL
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Just flatten
    tf.keras.layers.Dense(128, activation='relu'),  # One hidden layer
    tf.keras.layers.Dense(10, activation='softmax') # Output layer
])

# 5. COMPILE MODEL
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel ready! Starting training...")

# 6. TRAIN MODEL (Only 5 epochs for speed)
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# 7. TEST MODEL
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc:.2%}")

# 8. MAKE PREDICTIONS
predictions = model.predict(x_test[:10])
print("\nFirst 10 predictions:")
for i in range(5):
    predicted = np.argmax(predictions[i])
    actual = y_test[i]
    print(f"Image {i+1}: Predicted={predicted}, Actual={actual}")

# 9. SAVE MODEL
model.save('my_simple_model.h5')
print("\n✅ Model saved as 'my_simple_model.h5'")

# 10. PREDICT FROM UPLOADED IMAGE
from PIL import Image
import os

def predict_uploaded_image(image_path):
    """
    Predict digit from an uploaded image file.
    The image should contain a handwritten digit.
    """
    # Load the saved model
    loaded_model = tf.keras.models.load_model('my_simple_model.h5')
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: File '{image_path}' not found!")
        return None
    
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

# === HOW TO USE ===
print("\n" + "="*50)
print("📷 TO PREDICT YOUR OWN HANDWRITTEN DIGIT:")
print("="*50)
print("\n1. Save your handwritten digit image")
print("2. Run this command:")
print("\n   predict_uploaded_image('your_image.png')")
print("\n   Example:")
print("   predict_uploaded_image('my_digit.jpg')")
print("="*50)

# Uncomment the line below and replace with your image path to test:
# predict_uploaded_image('your_image_path_here.png')

print("\n🎉 PROJECT COMPLETE! 🎉")