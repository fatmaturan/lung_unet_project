import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset import split_data, create_dataset
from model import build_unet
from losses_metrics import (
    bce_dice_loss,
    dice_coef,
    iou,
    precision_m,
    recall_m
)

MODEL_PATH = "../models/best_lung_unet_model.h5"
print("Model yükleniyor...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "bce_dice_loss": bce_dice_loss,
        "dice_coef": dice_coef,
        "iou": iou,
        "precision_m": precision_m,
        "recall_m": recall_m
    }
)

print("Test verileri hazırlanıyor...")

x_train, x_val, x_test, y_train, y_val, y_test = split_data()

test_dataset = create_dataset(
    x_test,
    y_test,
    batch_size=1,
    augment=False,
    shuffle=False
)

y_true_all = []
y_pred_all = []

print("Tahminler yapılıyor...")

for images, masks in test_dataset:

    preds = model.predict(images, verbose=0)

    preds = (preds > 0.5).astype(np.uint8)

    y_true_all.extend(masks.numpy().flatten())
    y_pred_all.extend(preds.flatten())

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

cm = confusion_matrix(y_true_all, y_pred_all)

print("\n")
print("                Tahmin Edilen")
print("              Background     Lung")
print("Gerçek")
print(f"Background     {cm[0,0]}      {cm[0,1]}")
print(f"Lung           {cm[1,0]}      {cm[1,1]}")

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Background", "Lung"]
)

fig, ax = plt.subplots(figsize=(8,6))

disp.plot(
    cmap="Blues",
    values_format="d",
    ax=ax
)

plt.title("Lung Segmentation Confusion Matrix")

os.makedirs("../outputs", exist_ok=True)

plt.savefig(
    "../outputs/lung_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nKaydedildi:")
print("../outputs/lung_confusion_matrix.png")