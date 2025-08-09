import cv2

cap = cv2.VideoCapture(0)  # Open camera

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera Error!")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to close
        break

cap.release()
cv2.destroyAllWindows()
