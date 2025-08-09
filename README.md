# Sign Language to English Converter

## Overview

The **Sign Language to English Converter** is a real-time computer vision application that recognizes American Sign Language (ASL) hand gestures captured via webcam and converts them into English text. This tool aims to improve communication accessibility for deaf and mute individuals by enabling non-signers to understand sign language in everyday interactions.



## Features

- **Real-Time Hand Sign Detection:** Processes live video input from your webcam to identify ASL hand signs.
- **Deep Learning Model:** Uses a trained neural network to classify hand gestures with high accuracy.
- **English Text Output:** Displays the recognized sign as corresponding English letters or words.
- **Easy to Use:** Simple setup with Python scripts and standard libraries.
- **Extensible Dataset:** Comes with a base ASL image dataset; users can add new signs or collect more data.
- **Open Source:** Fully transparent and modifiable to suit custom needs.

---

## Technologies Used

- Python 3.x  
- TensorFlow / Keras — for building and loading the deep learning model  
- OpenCV — for webcam video capture and image preprocessing  
- NumPy — for numerical operations  
- Matplotlib (optional) — for visualization during training  
- OS and sys modules — for file management and system operations

---

## Project Structure

sign-language-converter/
│
├── asl_dataset/ # Folder containing subfolders of images for each ASL letter
│
├── sign_language_model.keras # Trained Keras model file for gesture recognition
│
├── real_time_detection.py # Main script to run live sign recognition with webcam
│
├── train_model.py # (Optional) Script for training the model from the dataset
│
├── requirements.txt # List of Python dependencies for the project
│
└── README.md # This detailed project documentation


---

## Installation Instructions

### Prerequisites

- Python 3.7 or newer installed on your system.
- Webcam connected and working.
- Git installed (for cloning the repository).
