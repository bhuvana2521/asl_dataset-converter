import os

dataset_path = "asl_dataset"  # Change this if needed
class_counts = {}

for letter in os.listdir(dataset_path):
    letter_path = os.path.join(dataset_path, letter)
    if os.path.isdir(letter_path):
        class_counts[letter] = len(os.listdir(letter_path))

print("Class distribution:")
for letter, count in sorted(class_counts.items()):
    print(f"{letter}: {count} images")
