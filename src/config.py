import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_IMAGE_DIR = os.path.join(BASE_DIR, "data", "raw", "images")
RAW_MASK_DIR = os.path.join(BASE_DIR, "data", "raw", "masks")

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_lung_unet_model.h5")

PLOT_DIR = os.path.join(BASE_DIR, "outputs", "plots")
PREDICTION_DIR = os.path.join(BASE_DIR, "outputs", "predictions")

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3

BATCH_SIZE = 2
EPOCHS = 30
LEARNING_RATE = 1e-4
SEED = 42

THRESHOLD = 0.5