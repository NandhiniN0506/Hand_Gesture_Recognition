**Hand Gesture Recognition**

Hand gesture recognition uses computer vision and machine learning to interpret human hand movements for intuitive interaction with digital devices. It has applications in virtual reality, sign language translation, and human-computer interaction.

**Table of Contents**

          * Project Overview
          * System Requirements
          * Dataset
          * Model Training
          * Evaluation
          * Acknowledgements
          
**Project Overview**

This project aims to develop a hand gesture recognition system using computer vision and machine learning techniques. The system is capable of recognizing various hand gestures in real-time, which can be used in different applications such as virtual reality, sign language translation, and human-computer interaction.

**System Requirements**

              * Python 3.7 or higher
              * OpenCV
              * NumPy
              * TensorFlow or PyTorch
              * Scikit-learn
              * Matplotlib
              * Installation
             


**Dataset**

The dataset should consist of images of hand gestures, organized in subdirectories named after the gesture classes (e.g., thumbs_up, thumbs_down, peace). Ensure that the dataset is placed in the data directory before running the preprocessing script.

Ensure that your dataset is placed in the data directory and organized in subdirectories by class labels.

Example: Directory Structure

data/
    thumbs_up/
        img1.jpg
        img2.jpg
        ...
    thumbs_down/
        img1.jpg
        img2.jpg
        ...


**Preprocessing**

import os
import cv2
import numpy as np

data_dir = 'data'
img_size = 64

def preprocess_data(data_dir, img_size):
    images = []
    labels = []
    classes = os.listdir(data_dir)
    
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    np.save('images.npy', images)
    np.save('labels.npy', labels)
    print(f'Data preprocessed and saved. Total samples: {len(images)}')

preprocess_data(data_dir, img_size)

**Model Training**

The model can be trained using either TensorFlow or PyTorch. The training script train.py handles the training process, including data loading, model training, and saving the trained model.

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

img_size = 64
num_classes = 2

# Load preprocessed data
images = np.load('images.npy')
labels = np.load('labels.npy')

# Normalize the images
images = images / 255.0

# Convert labels to one-hot encoding
labels = tf.keras.utils.to_categorical(labels, num_classes)

# Create a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(images, labels, epochs=10, batch_size=32, validation_split=0.2)

# Save the model
model.save('hand_gesture_model.h5')


**Evaluation**

The model can be evaluated using the evaluate.py script. This script loads the trained model and evaluates its performance on the test dataset, providing metrics such as accuracy, precision, recall, and F1-score.

               precision    recall  f1-score   support

      01_palm       1.00      1.00      1.00       366
         02_l       1.00      1.00      1.00       392
      03_fist       1.00      1.00      1.00       404
04_fist_moved       1.00      1.00      1.00       404
     05_thumb       1.00      1.00      1.00       403
     06_index       1.00      1.00      1.00       409
        07_ok       1.00      1.00      1.00       417
08_palm_moved       1.00      1.00      1.00       410
         09_c       1.00      1.00      1.00       418
      10_down       1.00      1.00      1.00       377

     accuracy                           1.00      4000
    macro avg       1.00      1.00      1.00      4000
 weighted avg       1.00      1.00      1.00      4000

Accuracy of the Model: 100.0%


**Acknowledgements**

Thanks to the open-source community for providing the tools and libraries used in this project.
Special thanks to the creators of the datasets used for training and evaluation.





