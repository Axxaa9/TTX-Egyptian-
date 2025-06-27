# preprocessing/preprocess_audio.py
import librosa
import numpy as np
import soundfile as sf
import os
from tqdm import tqdm
import pandas as pd

def wav_to_mel(wav_path, sr=22050, n_mels=80):
    y, _ = librosa.load(wav_path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.T  # (Time, Mel)

def preprocess_all(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    data = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        mel = wav_to_mel(row['wav_path'])
        np.save(os.path.join(output_dir, f"mel_{i}.npy"), mel)
        data.append((row['text'], f"mel_{i}.npy"))
    
    pd.DataFrame(data, columns=["text", "mel_file"]).to_csv(os.path.join(output_dir, "processed.csv"), index=False)
if __name__ == "__main__":
    preprocess_all(
        csv_path="data/arabic_egy_cleaned/cleaned.csv",
        output_dir="data/arabic_egy_cleaned/processed"
    )