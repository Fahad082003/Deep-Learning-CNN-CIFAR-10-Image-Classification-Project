📌 Project Overview

This project implements a Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset, which consists of 60,000 32x32 color images across 10 categories. The model was trained using TensorFlow/Keras and achieved 72% accuracy after 25 epochs.

📂 Dataset
Dataset: CIFAR-10 (Available in keras.datasets)

Classes:
Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

Size:
Training: 50,000 images
Testing: 10,000 images

🧠 Model Architecture
Input: 32x32x3 images
Convolutional layers with ReLU activation
MaxPooling layers for feature reduction
Fully connected Dense layers
Softmax output layer (10 classes)

⚙️ Training
Optimizer: Adam
Loss: Sparse Categorical Crossentropy
Epochs: 25 (best accuracy ~72% at epoch 25)

Batch size: 64
Early stopping can be used to prevent overfitting.

📊 Results
Accuracy: ~70% on test set
Visualization: First 10 test images displayed with Predicted vs True labels (green = correct, red = wrong).

🚀 How to Run
# Install dependencies
pip install tensorflow matplotlib numpy
# Train and evaluate
python cifar10_cnn.py

📸 Sample Output
(Green = Correct prediction, Red = Incorrect prediction)










(Green = Correct prediction, Red = Incorre
