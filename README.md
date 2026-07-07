# Lung Segmentation with U-Net

A deep learning project for lung segmentation from chest CT images using the U-Net architecture.

## Project Overview

This project performs semantic segmentation of lung regions using a U-Net model.

The pipeline includes:

- Image preprocessing
- Mask generation
- Model training
- Prediction
- Performance evaluation

---

## Dataset

Chest CT image dataset for lung segmentation.

Images and corresponding masks are used for supervised learning.

---

## Model

U-Net Architecture

Features:

- Encoder-Decoder Network
- Skip Connections
- Binary Semantic Segmentation

Loss Functions

- Binary Cross Entropy
- Dice Loss

Optimizer

- Adam

---

## Technologies

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

## Folder Structure

```
lung_unet_project
│
├── dataset/
├── models/
├── predictions/
├── notebooks/
├── train.py
├── predict.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## Results

The trained model accurately segments lung regions from CT images.

Metrics:

- Dice Score
- IoU
- Accuracy

---

## Future Improvements

- Attention U-Net
- Multi-class Segmentation
- nnU-Net
- Better Augmentation
- Transfer Learning

---

## Author

Fatma Turan
