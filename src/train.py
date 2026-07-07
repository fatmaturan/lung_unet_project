import os
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping

from config import MODEL_PATH, PLOT_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS
from dataset import split_data, create_dataset
from model import build_unet
from losses_metrics import bce_dice_loss, dice_coef, iou, precision_m, recall_m
from utils import create_dir, plot_history


def main():
    create_dir("models")
    create_dir(PLOT_DIR)

    x_train, x_val, x_test, y_train, y_val, y_test = split_data()

    print("Train görüntü sayısı:", len(x_train))
    print("Validation görüntü sayısı:", len(x_val))
    print("Test görüntü sayısı:", len(x_test))

    train_dataset = create_dataset(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        augment=True,
        shuffle=True
    )

    val_dataset = create_dataset(
        x_val,
        y_val,
        batch_size=BATCH_SIZE,
        augment=False,
        shuffle=False
    )

    model = build_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=bce_dice_loss,
        metrics=[dice_coef, iou, precision_m, recall_m]
    )

    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            save_best_only=True,
            monitor="val_loss",
            verbose=1
        ),
        CSVLogger("training_log_lung.csv"),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        )
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    plot_history(history, os.path.join(PLOT_DIR, "training_history_lung.png"))

    print("Eğitim tamamlandı.")
    print("En iyi model kaydedildi:", MODEL_PATH)


if __name__ == "__main__":
    main()