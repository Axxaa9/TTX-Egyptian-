# inference/utils.py
import librosa
import librosa.display
import numpy as np
import soundfile as sf

def mel_to_audio(mel_db, sr=22050):
    mel_db = mel_db.T
    mel = librosa.db_to_power(mel_db)
    audio = librosa.feature.inverse.mel_to_audio(mel, sr=sr, n_iter=60)
    return audio

def save_audio(audio, path, sr=22050):
    sf.write(path, audio, sr)
