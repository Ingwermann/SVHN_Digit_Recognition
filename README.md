# SVHN Digit Recognition with CNN
This project implements a Convolutional Neural Network (CNN) to classify digits (0–9) from the SVHN (Street View House Numbers) dataset. 
The model is built using Keras and TensorFlow, and achieves high accuracy with a lightweight custom architecture.


## Key Features
- Input: 32×32 grayscale images
- Model: Custom CNN (`cnn_model_2`)
  - 4 convolutional layers with LeakyReLU
  - BatchNormalization and Dropout
  - Final Dense layer with softmax
- Optimizer: Adam
- Loss: Categorical crossentropy
- Accuracy: ~93% (test set)

## Project Structure
svhn-digit-recognition/
├── models/
│ └── cnn_model_2.py # Model architecture
├── notebooks/
│ └── svhn_training_and_eval.ipynb # End-to-end training notebook
├── requirements.txt # Python dependencies
├── .gitignore
└── README.md

## Dataset
- **Name**: SVHN (Format 2 - Cropped Digits)
- **Size**:
  - Training: 42,000 images
  - Testing: 18,000 images
- **Note**: Convert to grayscale and normalize to [0,1] for training