# Date: April 13th
# Author: Saima Sanjida Shila

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, hidden_dim=512, num_layers=4, dropout=0.2, pad_token_id=3):
        super(TransformerLanguageModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, embed_dim))  # 512 = max_seq_len
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        embeds = self.embedding(input_ids) + self.pos_embedding[:, :seq_len, :]
        attention_mask = input_ids == self.pad_token_id
        out = self.transformer(embeds, src_key_padding_mask=attention_mask)
        logits = self.fc(out)
        return logits

    def generate(self, tokenizer, prompt, max_length=50, eos_token_id=None, temperature=1.0, device='cpu'):
        self.eval()
        input_ids = tokenizer.encode(prompt, out_type=int)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

        generated_ids = []
        for _ in range(max_length):
           logits = self.forward(input_tensor)  # Only logits, no hidden
           logits = logits[:, -1, :] / temperature
           probs = torch.nn.functional.softmax(logits, dim=-1)
           next_token_id = torch.multinomial(probs, num_samples=1).item()

           if eos_token_id is not None and next_token_id == eos_token_id:
             break

           generated_ids.append(next_token_id)
           input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]], device=device)], dim=1)

        return tokenizer.decode(generated_ids, out_type=str)
