import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
        else:
            raise ValueError("Loss is not implemented")

        total_error += np.sum(error)
        total_pairs += error.size

    if loss == "mae":
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




