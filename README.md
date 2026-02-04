# FER2013 Facial Emotion Recognition 🎭

This project is a Computer Vision application that detects human emotions from facial images using a Convolutional Neural Network (CNN). The model is trained on a simplified version of the FER2013 dataset and deployed with Streamlit and Hugging Face Spaces.

##  Project Overview

The goal of this project is to classify facial expressions into 3 emotion categories:

- Happy 🙂
- Sad 😢
- Surprise 😲

To speed up training and reduce complexity, the original FER2013 dataset was reduced to 3 classes.

The model is trained using deep learning techniques and deployed as an interactive web app.

##  Dataset

Dataset: FER2013  
Source: Kaggle Facial Emotion Recognition dataset

The dataset contains grayscale face images categorized by emotion.  
Images were resized and normalized before training.

##  Workflow

1. Dataset loading and inspection
2. Data preprocessing (resize + normalization)
3. Training/validation split
4. CNN model design
5. Model training
6. Performance visualization
7. Model comparison (v1 vs improved model)
8. Model saving (.h5)
9. Prediction on new images
10. Streamlit deployment

##  Model

A Convolutional Neural Network (CNN) was built using TensorFlow/Keras.

The second version of the model includes:

- Batch normalization
- Dropout regularization
- Improved stability during training

##  Results

The improved model achieved better validation accuracy and more stable learning curves compared to the baseline model.

##  Real-World Applications

This system can be used in:

- Human-computer interaction
- Mental health monitoring
- Smart cameras
- Customer emotion analytics
- Education and learning systems

## 🚀 Live Demo

Hugging Face Space:  
👉 **[(https://huggingface.co/spaces/abmias/fer2013-emotion-detection)]**

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy / Pandas
- Matplotlib / Seaborn
- Streamlit
- Hugging Face Spaces

##  Files

- `EmotionDetection.ipynb` → training notebook
- `app.py` → Streamlit app
- `emotion_model.h5` → trained model
- `requirements.txt` → dependencies
- `HFspace.txt` → demo link

---

✨ Built for Computer Vision coursework and portfolio development.
