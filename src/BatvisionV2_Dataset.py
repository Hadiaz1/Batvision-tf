import pandas as pd
import tensorflow as tf

import os
import yaml

from dataset_utils import _load_depth_file, _load_audio_file, min_max_normalize
from depth_transform import *
from audio_transform import *

def load_batvisionv2_dataset(params, version="train"):
    dataset_params = params["dataset"]
    transform_params = params["transform"]
    feature_extraction_params = params["audio_feature_extraction"]
    training_params = params["training"]

    df = dir_df(dataset_dir=dataset_params["dataset_dir"],
                annotation_file=f"{version}.csv",
                location_blacklist=dataset_params["location_blacklist"])

    def generator():
        for i in range(len(df.index)):
            row = df.iloc[i]
            depth_path = os.path.join(dataset_params["dataset_dir"], row['depth path'], row['depth file name'])
            audio_path = os.path.join(dataset_params["dataset_dir"], row['audio path'], row['audio file name'])

            # Depth
            depth = _load_depth_file(depth_path)
            depth = transform_depth(depth, params=transform_params)

            if transform_params["depth_norm"]:
                max_depth = transform_params["max_depth"]
                min_depth = 0.0
                depth = min_max_normalize(depth, min_depth, max_depth)
                depth = depth[..., tf.newaxis]

            # Audio
            waveform, _ = _load_audio_file(audio_path,
                                           sr=feature_extraction_params["sr"],
                                           max_depth=transform_params["max_depth"])
            if transform_params["max_depth"]:
                feature_extraction_params["win_length"] = 64
                feature_extraction_params["n_fft"] = 512
                feature_extraction_params["hop_length"] = 64 // 4

            spec = transform_audio(waveform, feature_extraction_params=feature_extraction_params).astype(np.float32)
            spec = spec[..., tf.newaxis]

            if transform_params["preprocess"] is not None and transform_params["preprocess"].lower() == "resize":
                depth = tf.image.resize(depth, [transform_params["image_size"], transform_params["image_size"]])
                spec = tf.image.resize(spec, [transform_params["image_size"], transform_params["image_size"]])

            yield spec, depth

    ds = tf.data.Dataset.from_generator(generator, output_types=(np.float32,np.float32))
    if version != "test":
        ds = ds.batch(training_params["batch_size"])
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def dir_df(dataset_dir, annotation_file, location_blacklist=None):
    location_list = os.listdir(dataset_dir)

    if location_blacklist:
        location_list = [location for location in location_list if location not in location_blacklist]

    location_csv_paths = [os.path.join(dataset_dir, location, annotation_file) for location in location_list]

    df_list = []
    for location_csv in location_csv_paths:
        df_list.append(pd.read_csv(location_csv))

    df_list = pd.concat(df_list)

    return df_list

if __name__ == "__main__":
    params = yaml.safe_load(open(r"C:\Users\CYTech Student\Batvision-tf\dataset_config.yaml"))
    ds = load_batvisionv2_dataset(params, "train")

    import matplotlib.pyplot as plt

    # Extract a batch of examples from the dataset
    num_images = 4
    batch = next(iter(ds.batch(num_images)))

    # Create subplots for displaying the images
    fig, axes = plt.subplots(nrows=1, ncols=num_images)

    # Plot each image in a subplot
    for i in range(num_images):
        spec, depth = batch[0][i], batch[1][i]
        axes[i].imshow(spec[:, :, 0])
        axes[i].set_title(f"Depth {i + 1}")

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()







