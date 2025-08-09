import os

dataset_path = "asl_dataset"

# Get class labels (folders inside asl_dataset)
class_labels = sorted([folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))])

print("Detected Class Labels:", class_labels)
