from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Concatenate, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model


def conv_block(inputs, filters, dropout_rate=0.0):
    x = Conv2D(filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)

    return x


def encoder_block(inputs, filters, dropout_rate=0.0):
    s = conv_block(inputs, filters, dropout_rate)
    p = MaxPooling2D((2, 2))(s)
    return s, p


def decoder_block(inputs, skip, filters, dropout_rate=0.0):
    x = Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip])
    x = conv_block(x, filters, dropout_rate)
    return x


def build_unet(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 32, 0.05)
    s2, p2 = encoder_block(p1, 64, 0.05)
    s3, p3 = encoder_block(p2, 128, 0.10)
    s4, p4 = encoder_block(p3, 256, 0.15)

    b1 = conv_block(p4, 512, 0.25)

    d1 = decoder_block(b1, s4, 256, 0.15)
    d2 = decoder_block(d1, s3, 128, 0.10)
    d3 = decoder_block(d2, s2, 64, 0.05)
    d4 = decoder_block(d3, s1, 32, 0.05)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    return Model(inputs, outputs, name="Lung_U-Net")