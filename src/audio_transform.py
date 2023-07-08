import librosa
import numpy as np
from librosa import feature


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



