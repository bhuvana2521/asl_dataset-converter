import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("sign_language_model.keras")
print("✅ Model Loaded Successfully!")

# Load class labels (Ensure the order matches the training dataset)
class_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", 
                "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z","0","1","2","3","4","5","6","7","8","9"]

print("✅ Class Labels Loaded:", class_labels)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for correct orientation
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract hand bounding box
            h, w, c = frame.shape
            x_min, y_min, x_max, y_max = w, h, 0, 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x, x_min), min(y, y_min)
                x_max, y_max = max(x, x_max), max(y, y_max)

            # Extract hand region
            hand_img = frame[y_min:y_max, x_min:x_max]

            # Check if the extracted hand image is valid
            if hand_img.shape[0] == 0 or hand_img.shape[1] == 0:
                continue  # Skip invalid frames

            # Resize and preprocess for model input
            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            hand_img = cv2.resize(hand_img, (64, 64))
            hand_img = np.expand_dims(hand_img, axis=-1)
            hand_img = np.expand_dims(hand_img, axis=0)
            hand_img = hand_img / 255.0

            # Predict sign
            prediction = model.predict(hand_img)
            predicted_index = np.argmax(prediction)
            confidence = prediction[0][predicted_index] * 100

            # Fix out-of-range index issue
            if predicted_index >= len(class_labels):
                print("❌ Error: Predicted index is out of range!")
                continue
            print("Raw Predictions:", prediction)


            predicted_label = class_labels[predicted_index]

            # Display result with confidence check
            if confidence > 85:  # Lowered threshold to 85%
                text = f"Prediction: {predicted_label} ({confidence:.2f}%)"
                color = (0, 255, 0)  
            else:
                text = "Uncertain Prediction"
                color = (0, 0, 255)

            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show the result
    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
