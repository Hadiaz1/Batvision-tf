from tensorflow.keras.layers import Conv2D, Concatenate, MaxPooling2D, Activation, BatchNormalization, Dropout, Conv2DTranspose, Input, UpSampling2D, Lambda, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
import yaml


FILTERS = [16, 32, 48, 64]

def get_model(params):

    if not params["transform"]["spec_to_rgb"]:
        raise ValueError("MobileNetv2 only accepts 3 input channels")

    input_shape = (params["transform"]["image_size"], params["transform"]["image_size"], 3)

    inputs = Input(shape=input_shape)
    encoder = MobileNetV2(input_tensor=inputs, input_shape=input_shape, weights="imagenet", include_top=False, alpha=0.35)

    skip_connection_names = ["input_1", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu" ]
    encoder_output_name = "block_13_expand_relu"

    encoder_output = encoder.get_layer(encoder_output_name).output
    x = encoder_output

    for i in range(1, len(skip_connection_names)+1):
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])

        x = Conv2D(FILTERS[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(FILTERS[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)
    model = Model(inputs=inputs, outputs= x)

    return model










