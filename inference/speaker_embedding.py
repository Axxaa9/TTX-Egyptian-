# inference/speaker_embedding.py
import numpy as np
import librosa

def extract_speaker_embedding(wav_paths, sr=22050, n_mels=80):
    embeddings = []
    for path in wav_paths:
        y, _ = librosa.load(path, sr=sr)
        mel = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel)
        mean_mel = np.mean(mel_db, axis=1)  # (n_mels,)
        embeddings.append(mean_mel)
    return np.mean(embeddings, axis=0)  # Average across reference audios
