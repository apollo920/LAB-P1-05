import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from src.transformer import Transformer
from src.attention import make_causal_mask
from scripts.dataset import load_translation_subset
from scripts.tokenizer import load_tokenizer, tokenize_pairs
from scripts.train import build_src_mask, build_tgt_mask


D_MODEL    = 128
NUM_HEADS  = 4
NUM_LAYERS = 2
D_FF       = 256
MAX_LEN    = 64
DROPOUT    = 0.1

OVERFIT_SAMPLES = 4
OVERFIT_EPOCHS  = 300
LR              = 1e-3
MAX_NEW_TOKENS  = 40


def autoregressive_decode(model, src_ids, tokenizer, pad_id, device,
                          max_new_tokens=MAX_NEW_TOKENS):
    """
    Loop auto-regressivo: gera tokens um a um ate encontrar
    o token <EOS> (SEP) ou atingir max_new_tokens.
    """
    model.eval()

    bos_id = tokenizer.cls_token_id
    eos_id = tokenizer.sep_token_id

    src = src_ids.unsqueeze(0).to(device)
    src_mask = build_src_mask(src, pad_id)

    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)

    generated = [bos_id]

    for _ in range(max_new_tokens):
        tgt_so_far = torch.tensor(
            [generated], dtype=torch.long, device=device
        )

        tgt_mask = build_tgt_mask(tgt_so_far, pad_id)

        with torch.no_grad():
            logits = model.decode(
                tgt_so_far, encoder_output,
                tgt_mask=tgt_mask, src_mask=src_mask
            )

        next_token_logits = logits[0, -1, :]
        next_token_id = next_token_logits.argmax(-1).item()

        generated.append(next_token_id)

        if next_token_id == eos_id:
            break

    clean_ids = [t for t in generated if t not in (bos_id, eos_id, pad_id)]
    decoded = tokenizer.decode(clean_ids, skip_special_tokens=True)
    return decoded


def run_overfit_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

    all_pairs  = load_translation_subset(100)
    mini_batch = all_pairs[:OVERFIT_SAMPLES]
    tokenizer  = load_tokenizer()
    pad_id     = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

    print("Frases usadas no overfitting test:")
    for i, p in enumerate(mini_batch):
        print(f"  [{i}] EN: {p['en']}")
        print(f"       DE: {p['de']}")
    print()

    src_list, tgt_list = tokenize_pairs(mini_batch, tokenizer, max_len=MAX_LEN)

    src_tensor = torch.stack(src_list).to(device)
    tgt_tensor = torch.stack(tgt_list).to(device)

    tgt_input  = tgt_tensor[:, :-1]
    tgt_target = tgt_tensor[:, 1:]

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

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Iniciando overfitting em {OVERFIT_SAMPLES} pares por {OVERFIT_EPOCHS} epocas...")
    print("=" * 55)

    src_mask = build_src_mask(src_tensor, pad_id)
    tgt_mask = build_tgt_mask(tgt_input, pad_id)

    for epoch in range(1, OVERFIT_EPOCHS + 1):
        model.train()

        logits = model(src_tensor, tgt_input,
                       src_mask=src_mask, tgt_mask=tgt_mask)

        vocab_size_out = logits.size(-1)
        loss = criterion(
            logits.reshape(-1, vocab_size_out),
            tgt_target.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:>4} | Loss: {loss.item():.6f}")

    print("=" * 55)
    print(f"\nLoss final: {loss.item():.6f}")

    print("\n--- Prova de Fogo: Geracao Auto-Regressiva ---\n")

    frase_idx = 0
    frase_src = mini_batch[frase_idx]["en"]
    frase_ref = mini_batch[frase_idx]["de"]

    print(f"Entrada (EN)    : {frase_src}")
    print(f"Referencia (DE) : {frase_ref}")

    src_single = src_list[frase_idx]
    traducao = autoregressive_decode(model, src_single, tokenizer, pad_id, device)

    print(f"Traducao gerada : {traducao}")
    print()

    ref_tokens = set(frase_ref.lower().split())
    gen_tokens = set(traducao.lower().split())
    overlap = ref_tokens & gen_tokens
    pct = 100 * len(overlap) / len(ref_tokens) if ref_tokens else 0

    print(f"Overlap de tokens com referencia: {len(overlap)}/{len(ref_tokens)} ({pct:.0f}%)")

    if pct >= 50:
        print("RESULTADO: Modelo memorizou o padrao com sucesso!")
    else:
        print("RESULTADO: Convergencia parcial — aumente OVERFIT_EPOCHS ou reduza LR.")


if __name__ == "__main__":
    run_overfit_test()
