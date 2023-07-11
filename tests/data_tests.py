from src.depth_transform import *
from src.audio_transform import *
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from src.dataset_utils import _load_audio_file

MAX_DEPTH = 10000
MIN_DEPTH = 0.0

def test_depth_arr(depth):
    params = {"max_depth": 10000}
    depth = transform_depth(depth, params)
    return depth

if __name__ == "__main__":
    #Image
    IMAGE_EXAMPLE = Image.open("camera_47.jpeg")

    #Depth
    DEPTH_EXAMPLE = np.load("depth_47.npy")
    DEPTH_EXAMPLE = test_depth_arr(DEPTH_EXAMPLE)
    DEPTH_IMG = Image.fromarray(DEPTH_EXAMPLE)


    #Audio
    params = {"feature_name": "spectrogram", "n_fft": 512, "power": 1, "win_length": 128, "hop_length": 64, "to_db": False}
    AUDIO_EXAMPLE, sr = _load_audio_file("audio_47.wav", sr=44100, max_depth=MAX_DEPTH)
    print(AUDIO_EXAMPLE.shape)
    AUDIO_SPEC = transform_audio(AUDIO_EXAMPLE, feature_extraction_params=params)
    print(AUDIO_SPEC.shape)

    fig, axs = plt.subplots(1, 3)

    im1 = axs[0].imshow(IMAGE_EXAMPLE)
    axs[0].axis('off')
    axs[0].set_title("Original Image")

    im2 = axs[1].imshow(DEPTH_IMG, vmin=0, vmax=1)
    axs[1].axis('off')
    axs[1].set_title("Depth")
    cbar2 = fig.colorbar(im2, ax=axs[1])


    im3 = axs[2].imshow(AUDIO_SPEC, aspect="auto", origin="lower")
    axs[2].set_title("Spectrogram")
    cbar3 = fig.colorbar(im2, ax=axs[2])

    plt.subplots_adjust(wspace=0.3)

    plt.show()

