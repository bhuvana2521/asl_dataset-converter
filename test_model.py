import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("sign_language_model.keras")

# Load class labels
class_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", 
                "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Generate a random test image (64x64 grayscale)
test_image = np.random.rand(1, 64, 64, 1)

# Predict
prediction = model.predict(test_image)
predicted_index = np.argmax(prediction)

print("Predicted Index:", predicted_index)
print("Prediction Confidence:", prediction[0][predicted_index])

# Check if index is in range
if predicted_index >= len(class_labels):
    print("❌ Error: Model is predicting an index out of range!")
else:
    print("✅ Predicted Label:", class_labels[predicted_index])
