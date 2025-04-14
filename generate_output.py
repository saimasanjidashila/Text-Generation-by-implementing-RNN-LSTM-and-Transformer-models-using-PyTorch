# generate.py

import torch
import argparse
import sentencepiece as spm

# Import your model class here   # or use LSTMLanguageModel, RNNLanguageModel
from RNNmodel import RNNLanguageModel
from LSTMmodel import LSTMLanguageModel
from transformerModel import TransformerLanguageModel
# === CONFIG ===
VOCAB_SIZE = 10000
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 4
PAD_TOKEN_ID = 3
MODEL_PATH = 'models/best_transformer_model.pt'
TOKENIZER_PATH = 'bpe_tokenizer.model'

def load_model(device):
    model = TransformerLanguageModel(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        pad_token_id=PAD_TOKEN_ID
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

def load_tokenizer():
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(TOKENIZER_PATH)
    return tokenizer

def main(prompt, max_length, temperature):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)
    tokenizer = load_tokenizer()

    output = model.generate(
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=max_length,
        eos_token_id=tokenizer.eos_id() if tokenizer.eos_id() != -1 else None,
        temperature=temperature,
        device=device
    )

    print("\n📜 Generated Text:")
    print(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a trained language model")
    parser.add_argument('--prompt', type=str, required=True, help='Initial prompt for text generation')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')

    args = parser.parse_args()
    main(args.prompt, args.max_length, args.temperature)
