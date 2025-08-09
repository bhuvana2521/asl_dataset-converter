import numpy as np
import cv2
import os

# Load the .npy file
npy_file = "sign_language_digits.npy"

# Check if the file exists
if not os.path.exists(npy_file):
    print("❌ Error: File 'sign_language_digits.npy' not found!")
    exit()

data = np.load(npy_file, allow_pickle=True)

# Create output folder for extracted images
output_folder = "sign_language_dataset"
os.makedirs(output_folder, exist_ok=True)

# Extract and save images
for i, (image, label) in enumerate(data):
    label_folder = os.path.join(output_folder, str(label))  # Create folder for each number
    os.makedirs(label_folder, exist_ok=True)  
    
    img_path = os.path.join(label_folder, f"{i}.jpg")
    cv2.imwrite(img_path, image)

print("✅ All images saved successfully in 'sign_language_dataset'!")
