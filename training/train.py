# training/train.py

import sys

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



# training/train.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from model.tts_model import TacotronLike
from model.loss import mel_loss
from training.utils import collate_fn

CSV_PATH = "data/arabic_egy_cleaned/processed/processed.csv"
NUM_EPOCHS = 100
BATCH_SIZE = 16
SAVE_PATH = "model.pth"
PRINT_EVERY = 10  # Print loss every N batches

class TTSDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        if self.df.empty:
            raise ValueError("CSV is empty!")
        print(f"✅ Dataset contains {len(self.df)} samples")

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        mel_path = os.path.join(os.path.dirname(CSV_PATH), self.df.iloc[idx]['mel_file'])
        if not os.path.exists(mel_path):
            raise FileNotFoundError(f"❌ Mel file missing: {mel_path}")
        return text, mel_path

    def __len__(self):
        return len(self.df)

def build_vocab(dataset):
    print("🔠 Building vocab...")
    all_text = ''.join(dataset.df['text'])
    unique_chars = sorted(set(all_text))
    vocab = {c: i + 1 for i, c in enumerate(unique_chars)}
    vocab['<pad>'] = 0
    print(f"✅ Vocab size: {len(vocab)}")
    return vocab

def train():
    print("🔥 Training started...")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"❌ Dataset CSV not found: {CSV_PATH}")

    dataset = TTSDataset(CSV_PATH)
    vocab = build_vocab(dataset)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, vocab)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    model = TacotronLike(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("✅ Model initialized")
    print(f"🚀 Starting training for {NUM_EPOCHS} epochs")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        print(f"\n📆 Epoch {epoch + 1}/{NUM_EPOCHS}")
        for batch_idx, (text_input, mel_target) in enumerate(loader):
            text_input = text_input.to(device)
            mel_target = mel_target.to(device)

            mel_input = mel_target[:, :-1, :]
            mel_target_shifted = mel_target[:, 1:, :]

            mel_out = model(text_input, mel_input)
            loss = mel_loss(mel_out, mel_target_shifted)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (batch_idx + 1) % PRINT_EVERY == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"🌀 Batch {batch_idx + 1}/{len(loader)} | Avg Loss: {avg_loss:.4f}")

        epoch_loss = total_loss / len(loader)
        print(f"✅ Epoch {epoch + 1} complete | Avg Loss: {epoch_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == NUM_EPOCHS:
            torch.save(model.state_dict(), f"model_epoch{epoch+1}.pth")
            print(f"💾 Model checkpoint saved: model_epoch{epoch+1}.pth")

    # Final save
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n🏁 Training finished. Final model saved as {SAVE_PATH}")

if __name__ == "__main__":
    train()
