# inference/synthesize.py

import torch
import numpy as np
from model.tts_model import TacotronLike
from inference.speaker_embedding import extract_speaker_embedding
from inference.utils import mel_to_audio, save_audio
from training.utils import pad_texts
import os

# ============== CONFIG ==============
MODEL_PATH = "model.pth"
REFERENCE_LIST = "inference/reference_list.txt"
VOCAB = {}  # Load same vocab used in training
OUTPUT_WAV = "output.wav"
# ====================================

def load_vocab_from_dataset(dataset_path="data/arabic_egy_cleaned/processed/processed.csv"):
    import pandas as pd
    df = pd.read_csv(dataset_path)
    all_text = ''.join(df['text'])
    vocab = {c: i+1 for i, c in enumerate(sorted(set(all_text)))}
    vocab['<pad>'] = 0
    return vocab

def synthesize(text):
    global VOCAB
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    VOCAB = load_vocab_from_dataset()
    model = TacotronLike(vocab_size=len(VOCAB))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    with open(REFERENCE_LIST, 'r') as f:
        ref_paths = [line.strip() for line in f.readlines()]
    speaker_embedding = extract_speaker_embedding(ref_paths)

    # Embed speaker info into decoder context (optional: modify model for this)
    text_tensor = pad_texts([text], VOCAB).to(device)
    mel_input = torch.zeros(1, 10, 80).to(device)  # Initial mel frame input (10 zeros)

    with torch.no_grad():
        mel_out = model(text_tensor, mel_input)
        mel_out = mel_out.squeeze().cpu().numpy()

    audio = mel_to_audio(mel_out)
    save_audio(audio, OUTPUT_WAV)
    print(f"✅ Speech saved to {OUTPUT_WAV}")

if __name__ == "__main__":
    text = input("Enter Arabic text to synthesize: ")
    synthesize(text)
