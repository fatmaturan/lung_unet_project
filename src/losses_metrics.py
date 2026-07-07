import tensorflow as tf

smooth = 1e-6


def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


def iou(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

    return (intersection + smooth) / (union + smooth)


def precision_m(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    true_positive = tf.reduce_sum(y_true * y_pred)
    predicted_positive = tf.reduce_sum(y_pred)

    return (true_positive + smooth) / (predicted_positive + smooth)


def recall_m(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    true_positive = tf.reduce_sum(y_true * y_pred)
    possible_positive = tf.reduce_sum(y_true)

    return (true_positive + smooth) / (possible_positive + smooth)