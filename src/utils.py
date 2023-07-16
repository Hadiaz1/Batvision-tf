import tensorflow as tf
import tensorflow.keras.backend as K

def get_saving_path(params):
    #Returns a saving path for the current experiment
    saving_path = params["general"]["output_dir"]
    saving_path += "/" + params["general"]["saved_dir"]
    saving_path += "/" + params["dataset"]["name"]

    feature_name = params["audio_feature_extraction"]["feature_name"]
    if feature_name == "melspectrogram":
        saving_path += "/" + params["audio_feature_extraction"]["feature_name"] + \
                         "_" + str(params["audio_feature_extraction"]["n_fft"]) +  "_" + str(params["audio_feature_extraction"]["n_mels"])
    elif feature_name == "spectrogram":
        saving_path += "/" + params["audio_feature_extraction"]["feature_name"] + \
                         "_" + str(params["audio_feature_extraction"]["n_fft"])
    else:
        raise ValueError("No such feature")

    saving_path += "_" + params["model"]["model_name"]


    if params["transform"]["preprocess"]:
        saving_path += "_img" + str(params["transform"]["image_size"])
    return saving_path


def custom_depth_loss(y_true, y_pred):
    w1, w2, w3 = 1.0, 3.0, 0.1

    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, 1.0)) * 0.5, 0, 1)

    return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))

def custom_depth_acc(y_true, y_pred):
  return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

