import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow.keras.backend as K
from dataset_utils import _load_audio_file, _load_depth_file
from BatvisionV2_Dataset import *
from audio_transform import *
from utils import *

def test_depth_arr(depth):
    params = {"max_depth": 10000}
    depth = transform_depth(depth, params)
    return depth

def load_model(path, custom_objects=None):
    return tf.keras.models.load_model(path, custom_objects=custom_objects)

def predict_depth(x, model):
    predictions = model.predict(x)
    return predictions

def compute_test_loss(actual, predictions, loss="mae"):
    total_error = 0
    total_pairs = 0

    for pred_img, actual_img in zip(predictions, actual):
        if loss == "mae":
            error = np.abs(pred_img - actual_img)
        elif loss == "rmse":
            error = np.square(pred_img - actual_img)
        elif loss == "custom_depth_loss":
            error = custom_depth_loss(actual_img, pred_img)
        else:
            raise ValueError("Loss is not implemented")

        total_error += np.sum(error)
        total_pairs += error.size

    if loss == "mae" or loss == "custom_depth_loss":
        loss = total_error / total_pairs
    elif loss == "rmse":
        loss = np.sqrt(total_error / total_pairs)
    return loss

def visualize_predictions(image, actual_depth, pred_depth):
    fig, axs = plt.subplots(1, 3)

    pred_depth = Image.fromarray(pred_depth)
    actual_depth = Image.fromarray(actual_depth)

    im1 = axs[0].imshow(image)
    axs[0].axis('off')
    axs[0].set_title("Original Image")

    im2 = axs[1].imshow(actual_depth, vmin=0, vmax=1)
    axs[1].axis('off')
    axs[1].set_title("Actual Depth")
    cbar2 = fig.colorbar(im2, ax=axs[1])

    im3 = axs[2].imshow(pred_depth, vmin=0, vmax=1)
    axs[2].axis('off')
    axs[2].set_title("Actual Depth")
    cbar3 = fig.colorbar(im3, ax=axs[2])

    plt.subplots_adjust(wspace=0.3)

    plt.show()


if __name__ == "__main__":
    # Image
    IMAGE_EXAMPLE = Image.open("tests/camera_47.jpeg")

    # Depth
    DEPTH_EXAMPLE = np.load("tests/depth_47.npy")
    DEPTH_EXAMPLE = test_depth_arr(DEPTH_EXAMPLE)
    DEPTH_IMG = Image.fromarray(DEPTH_EXAMPLE)

    # Audio
    params = {"feature_name": "spectrogram", "n_fft": 512, "power": 1, "win_length": 128, "hop_length": 64,
              "to_db": False}
    AUDIO_EXAMPLE, sr = _load_audio_file("tests/audio_47.wav", sr=44100, max_depth=10000)
    AUDIO_SPEC = transform_audio(AUDIO_EXAMPLE, feature_extraction_params=params)

    fig, axs = plt.subplots(1, 4)

    im1 = axs[0].imshow(IMAGE_EXAMPLE)
    axs[0].axis('off')
    axs[0].set_title("Original Image")

    im2 = axs[1].imshow(DEPTH_IMG, vmin=0, vmax=1)
    axs[1].axis('off')
    axs[1].set_title("Depth")
    cbar2 = fig.colorbar(im2, ax=axs[1])

    im3 = axs[2].imshow(AUDIO_SPEC, aspect="auto", origin="lower")
    axs[2].set_title("Spectrogram")
    cbar3 = fig.colorbar(im3, ax=axs[2])

    model = tf.keras.models.load_model("output/saved_models/batvisionv2/spectrogram_512_mobilenetv2_unet_img128.h5")
    AUDIO_SPEC = AUDIO_SPEC[..., tf.newaxis]

    AUDIO_SPEC = cv2.cvtColor(AUDIO_SPEC, cv2.COLOR_GRAY2RGB)
    AUDIO_SPEC = tf.image.resize(AUDIO_SPEC, [128, 128])
    AUDIO_SPEC = AUDIO_SPEC[tf.newaxis, ...]
    depth_pred = predict_depth(AUDIO_SPEC, model)
    depth_pred = np.squeeze(depth_pred, axis=0)
    depth_pred = np.squeeze(depth_pred, axis=-1)

    im4 = axs[3].imshow(depth_pred, vmin=0, vmax=0.25)
    axs[3].axis('off')
    axs[3].set_title("Depth")
    cbar4 = fig.colorbar(im4, ax=axs[3])

    plt.subplots_adjust(wspace=0.3)

    plt.show()



