import librosa
import numpy as np

def _load_audio_file(filepath, sr, max_depth=None):
    waveform , sr = librosa.load(filepath, sr=sr)

    if max_depth:
        cut = int((2 * max_depth / 340) * sr)
        waveform = waveform[:cut]

    return waveform, sr

def _load_depth_file(filepath):
    depth = np.load(filepath).astype(np.float32)

    return depth


def min_max_normalize(arr, min_value, max_value):
    normalized_array = (arr - min_value) / (max_value - min_value)

    return normalized_array


