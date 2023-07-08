import librosa
import numpy as np
from librosa import feature

def transform_audio(waveform, feature_extraction_params):
    if feature_extraction_params["feature_name"].lower()  == "spectrogram":
        spec = _get_spectrogram(waveform,
                                n_fft=feature_extraction_params["n_fft"],
                                power=feature_extraction_params["power"],
                                win_length=feature_extraction_params["win_length"],
                                hop_length=feature_extraction_params["hop_length"],
                                to_db=feature_extraction_params["to_db"])
    elif feature_extraction_params["feature_name"].lower() == "melspectrogram":
        spec = _get_mel_spectrogram(waveform,
                                    sr=feature_extraction_params["sr"],
                                    n_fft=feature_extraction_params["n_fft"],
                                    power=feature_extraction_params["power"],
                                    win_length=feature_extraction_params["win_length"],
                                    hop_length=feature_extraction_params["hop_length"],
                                    n_mels = feature_extraction_params["n_mels"],
                                    f_min = feature_extraction_params["f_min"],
                                    f_max = feature_extraction_params["f_max"],
                                    to_db = feature_extraction_params["to_db"])
    else:
        return ValueError("This feature is not implemented Yet! implemented feature are 'spectrogram' and 'melspectrogram' ")

    return spec

def _get_spectrogram(waveform, n_fft, power, win_length, hop_length, to_db):
    stft = librosa.stft(waveform, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    spectrogram = np.abs(stft) ** power

    if to_db:
        if power==1.0:
            db_spec = librosa.amplitude_to_db(spectrogram, ref=np.max)
        elif power==2.0:
            db_spec = librosa.power_to_db(spectrogram, ref=np.max)
        else:
            raise ValueError("Power must be either 1 or 0")
        return db_spec
    else:
        return np.log(spectrogram + 1e-4)


def _get_mel_spectrogram(waveform, sr, n_fft, power, win_length, hop_length,n_mels, f_min, f_max, to_db):
    mel_spec =  feature.melspectrogram(y=waveform,
                                       sr=sr,
                                       n_fft=n_fft,
                                       power=power,
                                       win_length=win_length,
                                       hop_length=hop_length,
                                       n_mels=n_mels,
                                       fmin=f_min,
                                       fmax=f_max)

    if to_db:
        if power==1.0:
            db_melspec = librosa.amplitude_to_db(mel_spec, ref=np.max)
        elif power==2.0:
            db_melspec = librosa.power_to_db(mel_spec, ref=np.max)
        else:
            raise ValueError("Power must be either 1 or 0")
        return db_melspec
    else:
        return np.log(mel_spec + 1e-4)



