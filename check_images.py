import os
import cv2

dataset_path = "asl_dataset"

for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    
    if os.path.isdir(label_path):  # Ensure it's a folder
        sample_images = [f for f in os.listdir(label_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

        if len(sample_images) == 0:
            print(f"⚠️ Folder {label} is empty or contains unsupported formats!")
        else:
            img_path = os.path.join(label_path, sample_images[0])
            img = cv2.imread(img_path)

            if img is None:
                print(f"❌ Error loading: {img_path}")
            else:
                print(f"✅ Successfully loaded: {img_path}")
                cv2.imshow("Sample Image", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
