# cat-dog-cnn-classifier
A CNN-based image classifier that tells cats apart from dogs, with model evaluation and a  interactive Streamlit app for real-time predictions.

## Cat vs Dog CNN Classifier

A Convolutional Neural Network (CNN) that classifies images as **Cat** or **Dog**, built from scratch with TensorFlow/Keras — complete with training, evaluation, and a  interactive web app for real-time predictions.

##  Features

- CNN trained on cat and dog images using data augmentation
- Accuracy/loss visualization and confusion matrix evaluation
- Saved, reusable trained model
- Interactive **Streamlit** web app — upload a photo, get an instant prediction
- Beginner-friendly notebook with a plain-language explanation before every step

##  Demo

Upload any cat or dog photo and the app instantly shows the prediction with a confidence score, in a clean dark-themed interface.

##  Project Structure

```
.
├── Dog_Cat_CNN_model.ipynb   # Training notebook (preprocessing → CNN → training → evaluation)
├── cdapp.py                    # Streamlit web app for live predictions
├── requirements.txt          # Project dependencies
└── README.md
```

##  Model Overview

A simple CNN with stacked Conv2D + MaxPooling layers, followed by Dense layers with dropout for regularization, trained with the Adam optimizer and binary cross-entropy loss. Early stopping is used to avoid overfitting.

##  Evaluation

Includes accuracy/loss curves across training epochs and a confusion matrix + classification report (precision, recall, F1-score) on the test set.

## streamlit application

#### locally run :
streamlit run cdapp.py

#### cloud deployement:
https://cat-dog-cnn-classifier-vsqbudwaqptewmsulahfbm.streamlit.app/

##  Technology used

- Python, TensorFlow / Keras
- NumPy, Matplotlib, Seaborn, scikit-learn
- Streamlit (deployment)

## Conclusion
This project successfully built a Convolutional Neural Network capable of distinguishing between cat and dog images with good accuracy. Starting from raw image data, the model learned to recognize visual patterns — shapes, textures, and features — that distinguish the two animals, without being explicitly told what to look for. The end-to-end pipeline, from data preprocessing to a live, interactive web app, demonstrates how a deep learning model can be taken from a research notebook to something a non-technical person can actually use.

## Key Insights

- Data augmentation helped the model generalize, not just memorize
- Dropout + early stopping kept overfitting in check
- The confusion matrix showed exactly where mistakes happened, beyond just accuracy
- A small CNN was enough for this task — no need for huge models
- Deployment (Streamlit) turned the model into something actually usable, not just a notebook experiment
