

def get_saving_path(params):
    #Returns a saving path for the current experiment
    saving_path = params["general"]["output"]
    saving_path += "/" + params["general"]["saved_dir"]
    saving_path += "/" + params["dataset"]["name"]

    feature_name = params["audio_feature_extraction"]["feature_name"]
    if feature_name == "melspectrogram":
        saving_path += "/" + params["audio_feature_extraction"]["feature_name"] + \
                         "_" + params["audio_feature_extraction"]["n_fft"] +  "_" + params["audio_feature_extraction"]["n_mels"]
    elif feature_name == "spectrogram":
        saving_path += "/" + params["audio_feature_extraction"]["feature_name"] + \
                         "_" + params["audio_feature_extraction"]["n_fft"]
    else:
        raise ValueError("No such feature")

    saving_path += "_" + params["model"]["model_name"]


    if params["transform"]["preprocess"]:
        saving_path += "_img" + params["transform"]["image_size"]
    return saving_path