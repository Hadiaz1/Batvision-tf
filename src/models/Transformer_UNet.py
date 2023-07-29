import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Conv2DTranspose, BatchNormalization, ReLU, Dropout, Add, \
    LayerNormalization, MultiHeadAttention, Embedding, Reshape, Concatenate


def mlp(x, mlp_dim, hidden_dim, dropout):
    x = Dense(mlp_dim, activation="gelu")(x)
    if dropout:
        x = Dropout(dropout)(x)
    x = Dense(hidden_dim)(x)
    if dropout:
        x = Dropout(dropout)(x)
    return x


def encoder(x, mlp_dim, num_heads, hidden_dim, dropout):
    skip_1 = x
    x = LayerNormalization()(x)
    x = MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim)(x, x)
    x = Add()([x, skip_1])

    skip_2 = x
    x = LayerNormalization()(x)
    x = mlp(x, mlp_dim=mlp_dim, hidden_dim=hidden_dim, dropout=dropout)
    x = Add()([x, skip_2])

    return x


def conv_block(x, num_filters, kernel_size=(3, 3)):
    x = Conv2D(num_filters, kernel_size=kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x


def up_conv_block(x, num_filters, kernel_size=(2, 2)):
    x = Conv2DTranspose(num_filters, kernel_size, padding="same", strides=2)(x)
    return x


def get_model(params):
    num_patches = (params["transform"]["image_size"] ** 2) // (params["model"]["patch_size"] ** 2)
    input_shape = (num_patches,
                   params["model"]["patch_size"] * params["model"]["patch_size"] * (
                       3 if params["transform"]["spec_to_rgb"] else 1))

    inputs = Input(shape=input_shape)

    patch_embed = Dense(params["model"]["hidden_dim"])(inputs)

    # Positional Embedding
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embed = Embedding(input_dim=num_patches, output_dim=params["model"]["hidden_dim"])(positions)

    x = patch_embed + pos_embed

    # Encoder
    skip_connections = []  # 3,6,9,12 according to the paper
    for i in range(1, params["model"]["num_layers"] + 1, 1):
        x = encoder(x, params["model"]["mlp_dim"], params["model"]["num_heads"], params["model"]["hidden_dim"],
                    params["model"]["dropout"])

        if i in [3, 6, 9, 12]:
            skip_connections.append(x)

    # Decoder
    z3, z6, z9, z12 = skip_connections
    size = params["transform"]["image_size"] // params["model"]["patch_size"]

    z0 = Reshape((params["transform"]["image_size"], params["transform"]["image_size"], 3 if params["transform"]["spec_to_rgb"] else 1))(inputs)
    z3 = Reshape((size, size, params["model"]["hidden_dim"]))(z3)
    z6 = Reshape((size, size, params["model"]["hidden_dim"]))(z6)
    z9 = Reshape((size, size, params["model"]["hidden_dim"]))(z9)
    z12 = Reshape((size, size, params["model"]["hidden_dim"]))(z12)

    x = up_conv_block(z12, 512, kernel_size=(2, 2))

    s = conv_block(up_conv_block(z9, 512, kernel_size=(2, 2)), 512, kernel_size=(3, 3))

    x = Concatenate()([x, s])
    x = conv_block(x, 512, kernel_size=(3, 3))
    x = conv_block(x, 512, kernel_size=(3, 3))
    x = up_conv_block(x, 256, kernel_size=(2, 2))

    t = conv_block(up_conv_block(z6, 256, kernel_size=(2, 2)), 256, kernel_size=(3, 3))
    t = conv_block(up_conv_block(t, 256, kernel_size=(2, 2)), 256, kernel_size=(3, 3))

    x = Concatenate()([x, t])
    x = conv_block(x, 256, kernel_size=(3, 3))
    x = conv_block(x, 256, kernel_size=(3, 3))
    x = up_conv_block(x, 128, kernel_size=(2, 2))

    u = conv_block(up_conv_block(z3, 128, kernel_size=(2, 2)), 128, kernel_size=(3, 3))
    u = conv_block(up_conv_block(u, 128, kernel_size=(2, 2)), 128, kernel_size=(3, 3))
    u = conv_block(up_conv_block(u, 128, kernel_size=(2, 2)), 128, kernel_size=(3, 3))

    x = Concatenate()([x, u])
    x = conv_block(x, 128, kernel_size=(3, 3))
    x = conv_block(x, 128, kernel_size=(3, 3))
    x = up_conv_block(x, 64, kernel_size=(2, 2))

    v = conv_block(z0, 64, kernel_size=(3, 3))
    v = conv_block(v, 64, kernel_size=(3, 3))

    x = Concatenate()([x, v])

    x = conv_block(x, 64, kernel_size=(3, 3))
    x = conv_block(x, 64, kernel_size=(3, 3))

    outputs = Conv2D(1, kernel_size=(1, 1), padding="same", activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
