# Date: April 12th
# Author: Saima Sanjida Shila

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.2, pad_token_id=3):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, hidden=None):
        embeds = self.embedding(input_ids)
        output, hidden = self.lstm(embeds, hidden)
        logits = self.fc(output)
        return logits, hidden

    def generate(self, tokenizer, prompt, max_length=50, eos_token_id=None, temperature=1.0, device='cpu'):
        self.eval()
        input_ids = tokenizer.encode(prompt, out_type=int)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

        generated_ids = []
        hidden = None

        for _ in range(max_length):
          logits, hidden = self.forward(input_tensor, hidden)
          # Adjust with temperature
          logits = logits[:, -1, :] / temperature  
          probs = F.softmax(logits, dim=-1)

          # Sample from probability distribution instead of picking argmax
          next_token_id = torch.multinomial(probs, num_samples=1).item()

          if eos_token_id is not None and next_token_id == eos_token_id:
             break

          generated_ids.append(next_token_id)
          input_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

        return tokenizer.decode(generated_ids, out_type=str)
