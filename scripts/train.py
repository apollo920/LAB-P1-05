import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from src.transformer import Transformer
from src.attention import make_causal_mask
from scripts.dataset import load_translation_subset
from scripts.tokenizer import load_tokenizer, build_dataloader


D_MODEL    = 128
NUM_HEADS  = 4
NUM_LAYERS = 2
D_FF       = 256
MAX_LEN    = 64
DROPOUT    = 0.1

BATCH_SIZE  = 32
NUM_EPOCHS  = 15
LR          = 1e-4
NUM_SAMPLES = 1000


def build_src_mask(src, pad_id):
    mask = (src != pad_id).unsqueeze(1).unsqueeze(2)
    return mask.int()


def build_tgt_mask(tgt, pad_id):
    seq_len = tgt.size(1)
    device  = tgt.device

    pad_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2).int()
    causal   = make_causal_mask(seq_len, device=device).int()

    return pad_mask * causal


def train_one_epoch(model, loader, criterion, optimizer, pad_id, device):
    model.train()
    total_loss   = 0.0
    total_tokens = 0

    for src, tgt_input, tgt_target in loader:
        src        = src.to(device)
        tgt_input  = tgt_input.to(device)
        tgt_target = tgt_target.to(device)

        src_mask = build_src_mask(src, pad_id)
        tgt_mask = build_tgt_mask(tgt_input, pad_id)

        # Forward pass
        logits = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)

        vocab_size = logits.size(-1)
        logits_flat     = logits.reshape(-1, vocab_size)
        tgt_target_flat = tgt_target.reshape(-1)

        # Loss
        loss = criterion(logits_flat, tgt_target_flat)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        num_tokens   = (tgt_target_flat != pad_id).sum().item()
        total_loss  += loss.item() * num_tokens
        total_tokens += num_tokens

    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

    pairs     = load_translation_subset(NUM_SAMPLES)
    tokenizer = load_tokenizer()
    loader, pad_id = build_dataloader(pairs, tokenizer,
                                      batch_size=BATCH_SIZE,
                                      max_len=MAX_LEN)

    vocab_size = tokenizer.vocab_size
    print(f"\nVocabulario: {vocab_size} tokens | PAD id: {pad_id}\n")

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametros treinaveis: {num_params:,}\n")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)

    print("=" * 55)
    print(f"{'Epoch':>6}  {'Loss':>10}")
    print("=" * 55)

    loss_history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss = train_one_epoch(model, loader, criterion, optimizer, pad_id, device)
        loss_history.append(avg_loss)
        print(f"{epoch:>6}  {avg_loss:>10.4f}")

    print("=" * 55)
    queda = loss_history[0] - loss_history[-1]
    print(f"\nLoss inicial : {loss_history[0]:.4f}")
    print(f"Loss final   : {loss_history[-1]:.4f}")
    print(f"Queda total  : {queda:.4f} ({100 * queda / loss_history[0]:.1f}%)\n")

    save_path = os.path.join(os.path.dirname(__file__), "..", "model_checkpoint.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "pad_id": pad_id,
        "loss_history": loss_history,
    }, save_path)
    print(f"Checkpoint salvo em: {save_path}")

    return model, tokenizer, pad_id, pairs, loss_history


if __name__ == "__main__":
    run_training()