# preprocessing/build_dataset.py
import os
import pandas as pd
from text_cleaner import clean_arabic_text

def build_dataset(metadata_path, wav_dir, output_csv):
    df = pd.read_csv(metadata_path, sep='|', header=None, names=['file', 'text'])
    cleaned_data = []
    for _, row in df.iterrows():
        fname = row['file'].strip()
        text = clean_arabic_text(row['text'])
        path = os.path.join(wav_dir, fname)
        if os.path.exists(path):
          cleaned_data.append([path, text])
        else:
            print(f"Missing: {path}")

    pd.DataFrame(cleaned_data, columns=['wav_path', 'text']).to_csv(output_csv, index=False)
    print(f"Preprocessed {len(cleaned_data)} entries.")
if __name__ == "__main__":
    build_dataset(
        metadata_path="data/arabic_egy_cleaned/metadata.csv",
        wav_dir="data/arabic_egy_cleaned/wavs",
        output_csv="data/arabic_egy_cleaned/cleaned.csv"
    )
