import cv2
import numpy as np
import tensorflow as tf

# Load class labels
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Load trained model
model = tf.keras.models.load_model("sign_language_model.keras")
print("✅ Model loaded successfully.")

# Read test image
image_path = "test_image.jpeg"  # Make sure this path is correct
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image loaded successfully
if image is None:
    print("❌ ERROR: Test image not found! Check the path.")
    exit()

print("✅ Test image loaded successfully.")

# Resize image to match model input size (64x64)
image = cv2.resize(image, (64, 64))

# Normalize pixel values to range [0, 1]
image = image / 255.0

# Expand dimensions to fit model input shape
input_data = np.expand_dims(image, axis=(0, -1))  # Shape: (1, 64, 64, 1)

# Debugging: Check input data
print(f"\n📏 Input shape: {input_data.shape}")
print(f"🔍 Sample pixel values: {input_data[0][0]}")

# Predict
print("\n🤖 Making prediction...")
prediction = model.predict(input_data)
print(f"📊 Raw Prediction Values: {prediction}")

# Get top prediction
predicted_index = np.argmax(prediction)
predicted_label = class_labels[predicted_index]
confidence = prediction[0][predicted_index] * 100  # Convert to percentage

# Get top 3 predictions
top_3_indices = np.argsort(prediction[0])[-3:][::-1]
top_3_labels = [class_labels[i] for i in top_3_indices]
top_3_confidences = [prediction[0][i] * 100 for i in top_3_indices]

# Print results
print(f"\n✅ Predicted Letter: {predicted_label} (Confidence: {confidence:.2f}%)")
print(f"📌 Top 3 Predictions:")
for i, (label, conf) in enumerate(zip(top_3_labels, top_3_confidences), 1):
    print(f"  {i}. {label} - {conf:.2f}%")

# Display test image
cv2.imshow("Test Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
