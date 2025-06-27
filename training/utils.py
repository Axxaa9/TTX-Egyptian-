# training/utils.py

import torch
import numpy as np

def pad_texts(texts, vocab):
    max_len = max(len(t) for t in texts)
    padded = np.zeros((len(texts), max_len), dtype=int)
    for i, t in enumerate(texts):
        padded[i, :len(t)] = [vocab[c] for c in t]
    return torch.LongTensor(padded)

def collate_fn(batch, vocab):
    texts = [b[0] for b in batch]
    mels = [np.load(b[1]) for b in batch]

    text_tensor = pad_texts(texts, vocab)
    mel_tensor = torch.nn.utils.rnn.pad_sequence(
        [torch.Tensor(m) for m in mels], batch_first=True
    )
    return text_tensor, mel_tensor
