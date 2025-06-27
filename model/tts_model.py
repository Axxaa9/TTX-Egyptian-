# model/tts_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        return output  # shape: (B, T, 2*hidden_dim)

class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.query = nn.Linear(dec_dim, enc_dim)
        self.energy = nn.Linear(enc_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        query = self.query(decoder_hidden).unsqueeze(1)
        scores = torch.tanh(encoder_out + query)
        attn_weights = F.softmax(self.energy(scores), dim=1)
        context = torch.sum(attn_weights * encoder_out, dim=1)
        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, enc_dim, mel_dim=80, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(enc_dim + mel_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, mel_dim)

    def forward(self, context, mel_input):
        # context: (B, enc_dim), mel_input: (B, 1, mel_dim)
        context = context.unsqueeze(1).expand(-1, mel_input.size(1), -1)
        x = torch.cat([context, mel_input], dim=-1)
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out

class TacotronLike(nn.Module):
    def __init__(self, vocab_size, mel_dim=80):
        super().__init__()
        self.encoder = Encoder(vocab_size)
        self.attention = Attention(enc_dim=512, dec_dim=256)
        self.decoder = Decoder(enc_dim=512, mel_dim=mel_dim)

    def forward(self, text_input, mel_input):
        encoder_out = self.encoder(text_input)
        decoder_hidden = torch.zeros(text_input.size(0), 256).to(text_input.device)

        context, _ = self.attention(encoder_out, decoder_hidden)
        mel_out = self.decoder(context, mel_input)
        return mel_out
