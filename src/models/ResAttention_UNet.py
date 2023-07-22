from tensorflow.keras.layers import Input, Conv2D, Concatenate, BatchNormalization, Add, Dropout, Activation, UpSampling2D, \
    Multiply, MaxPooling2D
from tensorflow.keras.models import Model
import yaml

def residual_block(x, out_filters, kernel_size, dropout, batch_norm=False):
    inputs = x

    x = Conv2D(out_filters, kernel_size=kernel_size, padding="same")(inputs)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(out_filters, kernel_size=kernel_size, padding="same")(x)
    if batch_norm:
        x = BatchNormalization()(x)

    if dropout > 0:
        x = Dropout(dropout)(x)

    shortcut = Conv2D(out_filters, kernel_size=(1, 1), padding="same")(inputs)
    if batch_norm:
        shortcut = BatchNormalization()(shortcut)

    x = Add()([shortcut, x])
    x = Activation("relu")(x)

    return x

def gating_signal(x, out_size, batch_norm=False):
    x = Conv2D(out_size, kernel_size=(1, 1), padding="same")
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def attention_block(x, gating):
    x_shape = x.get_shape()
    out_filters = x_shape[-1]
    g_shape = gating.get_shape()

    inputs = x

    x = Conv2D(out_filters, kernel_size=(1, 1), strides=(2, 2), padding="same")(inputs)

    gating = Conv2D(out_filters, kernel_size=(1, 1), strides=(1, 1), padding="same")(gating)

    x = Add()([x, gating])

    x = Activation("relu")(x)

    x = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)

    x = Activation("sigmoid")(x)

    x = UpSampling2D(size=(x_shape[1] // x.get_shape()[1], x_shape[2] // x.get_shape()[2]))(x)

    x = Multiply()([x, inputs])

    x = Conv2D(out_filters, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)

    return x

# print x_shape and make sure it doesnt print NOne for batch size ro soemthing

def get_model(params):
    KERNEL_SIZE = (3, 3)
    UP_SAMP_SIZE = 2
    N_FILTERS = 64
    BATCH_NORM = True
    DROPOUT = params["model"]["dropout"] if params["model"]["dropout"] is not None else 0.0

    input_shape = (params["transform"]["image_size"],
                   params["transform"]["image_size"],
                   3 if params["transform"]["spec_to_rgb"] else 1)

    # Encoder
    inputs = Input(shape=input_shape)

    conv_1 = residual_block(inputs, N_FILTERS, KERNEL_SIZE, DROPOUT, BATCH_NORM)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)

    conv_2 = residual_block(pool_1, 2 * N_FILTERS, KERNEL_SIZE, DROPOUT, BATCH_NORM)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)

    conv_3 = residual_block(pool_2, 4 * N_FILTERS, KERNEL_SIZE, DROPOUT, BATCH_NORM)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)

    conv_4 = residual_block(pool_3, 8 * N_FILTERS, KERNEL_SIZE, DROPOUT, BATCH_NORM)
    pool_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

    conv_5 = residual_block(pool_4, 16 * N_FILTERS, KERNEL_SIZE, DROPOUT, BATCH_NORM)

    # Decoder
    att_1 = attention_block(conv_4, conv_5)
    conv_up_1 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE))(conv_5)
    up_1 = Concatenate()([conv_up_1, att_1])
    conv_6 = residual_block(up_1, 8 * N_FILTERS, KERNEL_SIZE, DROPOUT, BATCH_NORM)

    att_2 = attention_block(conv_3, conv_6)
    conv_up_2 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE))(conv_6)
    up_2 = Concatenate()([conv_up_2, att_2])
    conv_7 = residual_block(up_2, 4 * N_FILTERS, KERNEL_SIZE, DROPOUT, BATCH_NORM)

    att_3 = attention_block(conv_2, conv_7)
    conv_up_3 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE))(conv_7)
    up_3 = Concatenate()([conv_up_3, att_3])
    conv_8 = residual_block(up_3, 2 * N_FILTERS, KERNEL_SIZE, DROPOUT, BATCH_NORM)

    att_4 = attention_block(conv_1, conv_8)
    conv_up_4 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE))(conv_8)
    up_4 = Concatenate()([conv_up_4, att_4])
    conv_8 = residual_block(up_4, 2 * N_FILTERS, KERNEL_SIZE, DROPOUT, BATCH_NORM)

    conv_last = Conv2D(1, kernel_size=(1, 1), padding="same")(conv_8)
    outputs = BatchNormalization()(conv_last)
    outputs = Activation("sigmoid")(outputs)
    model = Model(inputs=inputs, outputs=outputs)

    return model
