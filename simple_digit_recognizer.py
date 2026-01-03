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
print("\n🎉 PROJECT COMPLETE! 🎉")