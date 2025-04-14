# Date: April 12th
# Author: Saima Sanjida Shila

# IMPORT NECESSARY
import json
import torch
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tranformerModel import TransformerLanguageModel
import matplotlib.pyplot as plt
import torch.optim as optim
import sentencepiece as spm
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
nltk.download('punkt')
from nltk.tokenize import wordpunct_tokenize

# CONFIG
EPOCHS = 2
LEARNING_RATE = 0.0001
BATCH_SIZE = 128
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 4
MAX_SEQ_LEN = 64
PAD_TOKEN_ID = 3
# LOAD TRAIN, TEST DATASET
TRAIN_FILE = "data/train_fixed.jsonl"
VAL_FILE = "data/test_fixed.jsonl"
TOKENIZER_PATH = "bpe_tokenizer.model"
# PYTORCH DATALOADER
class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_seq_len=128):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                text = item["prompt"] + " " + item["completion"]
                token_ids = tokenizer.encode(text, out_type=int)[:max_seq_len]
                if len(token_ids) < 2:
                    continue
                self.samples.append(token_ids)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, target_ids

def collate_fn(batch):
    input_batch, target_batch = zip(*batch)
    input_batch = nn.utils.rnn.pad_sequence(input_batch, batch_first=True, padding_value=PAD_TOKEN_ID)
    target_batch = nn.utils.rnn.pad_sequence(target_batch, batch_first=True, padding_value=PAD_TOKEN_ID)
    return input_batch, target_batch

def load_tokenizer(path):
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp

# Training
def train_model():
    # Temporarily disable MPS to avoid bus error
    device = torch.device("cpu")  
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    vocab_size = tokenizer.get_piece_size()
    # Load full dataset once
    full_dataset = TextDataset(TRAIN_FILE, tokenizer, MAX_SEQ_LEN)
    # Split samples BETWEEN TRAIN AND VAL
    train_samples = full_dataset.samples[:30000]
    val_samples = full_dataset.samples[-10000:]
    # Wrap with same class to preserve __getitem__ behavior
    class SubsetTextDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            tokens = self.samples[idx]
            input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
            target_ids = torch.tensor(tokens[1:], dtype=torch.long)
            return input_ids, target_ids

    train_dataset = SubsetTextDataset(train_samples)
    val_dataset = SubsetTextDataset(val_samples)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    # TRAIN MODEL ON CONFIG
    model = TransformerLanguageModel(vocab_size, embed_dim=EMBED_DIM, 
                                     hidden_dim=HIDDEN_DIM, 
                                     num_layers=NUM_LAYERS, 
                                     pad_token_id=PAD_TOKEN_ID).to(device)
    # APPLIED ADAM OPTIMIZER
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    #APPLIED SCHEDULED LEARNING RATE
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    best_val_loss = float('inf')
    early_stop_counter = 0
    # EARLY stop if no improvement for 3 epochs
    early_stop_patience = 3  
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        for input_ids, target_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Train Loss: {avg_train_loss:.4f}")
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Val Loss: {avg_val_loss:.4f}")
        val_losses.append(avg_val_loss)

        perplexity = torch.exp(torch.tensor(avg_val_loss))
        # PERPLEXITY FOR EACH EPOCH ON TRAINING 
        print(f"Perplexity: {perplexity:.2f}")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
          best_val_loss = avg_val_loss
          torch.save(model.state_dict(), "best_transformer_model.pt")
          print(" Model saved!")
          early_stop_counter = 0  
        else:
          early_stop_counter += 1
          print(f"⚠️ No improvement. Early stop counter: {early_stop_counter}/3")
          if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break
    # SAVE THE TRAINED MODEL
    model.load_state_dict(torch.load("best_transformer_model.pt"))
    # LOAD TEST DATASET
    evaluate_on_test(model, tokenizer, "data/test_fixed.jsonl", device)
    prompt = "<bos>Which do you prefer? Dogs or cats"
    # TEST MODEL ON GIVEN PROMPT
    output = model.generate(tokenizer, prompt, eos_token_id=2, temperature=0.8, device=device)
    print("Generated:", output)

    # Confirm SAVED PNG DIRECTORY
    print("Saving plot to:", os.getcwd())
    # Plot and save loss curve
    if train_losses and val_losses:
      plt.figure(figsize=(8, 5))
      plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
      plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.title('Training and Validation Loss Curve')
      plt.legend()
      plt.grid(True)
      plt.tight_layout()
      plt.savefig("loss_transformer_curve.png")
      plt.close()
      print("loss_transformer_curve.png saved.")
    else:
      print(" train_losses or val_losses is empty. No plot generated.")

def evaluate_on_test(model, tokenizer, test_file, device):
    print("\n Evaluating on test set...")
    test_dataset = TextDataset(test_file, tokenizer, MAX_SEQ_LEN)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, reduction='sum')
    bleu_scores = []
    smoother = SmoothingFunction()
    with torch.no_grad():
        for input_ids, target_ids in tqdm(test_loader, desc="Test Eval"):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            # Perplexity
            logits = model(input_ids)  
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            total_loss += loss.item()
            total_tokens += (target_ids != PAD_TOKEN_ID).sum().item()
            # BLEU score
            decoded_input = tokenizer.decode(input_ids[0].tolist(), out_type=str)
            decoded_target = tokenizer.decode(target_ids[0].tolist(), out_type=str)
            generated_text = model.generate(
                tokenizer,
                decoded_input,
                eos_token_id=2,
                max_length=target_ids.shape[1] + 10,
                device=device
            )
            reference = wordpunct_tokenize(decoded_target)
            candidate = wordpunct_tokenize(generated_text)
            score = sentence_bleu([reference], candidate, smoothing_function=smoother.method1)
            bleu_scores.append(score)
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"Test Perplexity: {perplexity:.2f}")
    print(f"Average BLEU Score: {avg_bleu:.4f}")

# MAIN HERE
if __name__ == "__main__":
    train_model()
