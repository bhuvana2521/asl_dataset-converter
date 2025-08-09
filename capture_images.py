import cv2
import mediapipe as mp
import os

# Ask user for the letter/word they want to capture
label = input("Enter the letter/word you want to capture (e.g., A, B, hello): ").strip()
folder_path = "sign_language_dataset"

# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Open the webcam
cap = cv2.VideoCapture(0)

i = 0  # Image counter

while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        print("Error: Could not capture frame.")
        break

    # Convert frame to RGB (Mediapipe needs RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Draw hand landmarks if detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the letter/word on the screen
    cv2.putText(frame, f"Sign: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Capture Images", frame)

    # Press 's' to save the image
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        img_path = os.path.join(folder_path, f"{label}_{i}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"✅ Saved: {img_path}")
        i += 1  # Increase counter

    # Press 'q' to exit
    elif key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
