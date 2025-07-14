Deep Learning Image Classification

Company Name: CODTECH IT SOLUTIONS

Name: Geebu Pavani

ID: CT04DH2208

Domain: Data Science

Duration: July 10th, 2025 to August 10th, 2025

Mentor: Neela Santhosh Kumar

📝 Project Description:
This repository contains the solution for Task 2: Deep Learning - Image Classification as part of the CodTech Internship. The goal of this task was to develop and train a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset.

CIFAR-10 is a well-known benchmark dataset consisting of 60,000 32x32 color images across 10 distinct classes (such as airplane, automobile, bird, cat, etc.). This task required building a deep learning model, training it on the dataset, evaluating its performance, and saving it for future use.

🔧 CNN Workflow:
✅ 1. Data Loading & Preprocessing:
The CIFAR-10 dataset was loaded using tensorflow_datasets, and images were normalized (pixel values scaled between 0 and 1). The data was batched, shuffled, and prefetched for performance optimization.

✅ 2. Model Building (CNN):
A custom Convolutional Neural Network (CNN) was built using the Keras Sequential API. It included multiple convolutional layers, max pooling, a flatten layer, and two dense layers with ReLU and softmax activations.

✅ 3. Model Training:
The model was compiled with Adam optimizer and sparse categorical cross-entropy loss. It was trained over 10 epochs using both training and validation datasets.

✅ 4. Evaluation & Visualization:
The training performance was tracked and visualized through accuracy and loss graphs. A confusion matrix and classification report were generated using sklearn to evaluate the model’s predictions on test data.

✅ 5. Model Saving & Loading:
The trained model was saved in .keras format inside the models/ directory and successfully reloaded later to confirm persistence and usability.

📂 Files Included:
main.ipynb – Complete Jupyter notebook containing code for model building, training, evaluation, and saving

models/ – Folder where the trained CNN model is saved (cifar10_model.keras)

Documentation.md – This file describing the project

requirements.txt – Python dependencies used for this task

🧠 Key Concepts Practiced:
Deep learning with Convolutional Neural Networks (CNNs)

Image classification using CIFAR-10

Model training, evaluation, and performance visualization

Confusion matrix and classification reporting

Saving and reloading deep learning models

🛠 Tools & Libraries Used:
Python 3.12

TensorFlow

TensorFlow Datasets

Matplotlib

NumPy

Seaborn

Scikit-learn

Jupyter Notebook

Visual Studio Code

Git & GitHub

🚀 Outcome:
This task demonstrates the ability to build and train an end-to-end deep learning model for real-world image classification. It shows proficiency in TensorFlow/Keras, data preprocessing techniques, performance evaluation, and model serialization. The final model achieved approximately 71% accuracy, and is reusable for inference or further fine-tuning.


