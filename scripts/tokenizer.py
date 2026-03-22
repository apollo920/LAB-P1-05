import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


TOKENIZER_NAME = "bert-base-multilingual-cased"
MAX_SEQ_LEN = 64
PAD_ID = 0


def load_tokenizer():
    print(f"Carregando tokenizador: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return tokenizer


def tokenize_pairs(pairs, tokenizer, max_len=MAX_SEQ_LEN):
    src_ids_list = []
    tgt_ids_list = []

    for pair in pairs:
        src_encoded = tokenizer(
            pair["en"],
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        src_ids = src_encoded["input_ids"].squeeze(0)  
        tgt_tokens = tokenizer.encode(
            pair["de"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_len - 2,     
        )

        bos_id = tokenizer.cls_token_id   
        eos_id = tokenizer.sep_token_id   
        pad_id = tokenizer.pad_token_id   

        tgt_with_special = [bos_id] + tgt_tokens + [eos_id]

       
        padding_len = max_len - len(tgt_with_special)
        tgt_padded = tgt_with_special + [pad_id] * padding_len

        tgt_ids = torch.tensor(tgt_padded, dtype=torch.long) 

        src_ids_list.append(src_ids)
        tgt_ids_list.append(tgt_ids)

    return src_ids_list, tgt_ids_list


class TranslationDataset(Dataset):
    def __init__(self, src_ids_list, tgt_ids_list):
        assert len(src_ids_list) == len(tgt_ids_list)
        self.src = src_ids_list
        self.tgt = tgt_ids_list

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx]                 
        tgt_full = self.tgt[idx]            

        tgt_input = tgt_full[:-1]           
        tgt_target = tgt_full[1:]           

        return src, tgt_input, tgt_target


def build_dataloader(pairs, tokenizer, batch_size=32, max_len=MAX_SEQ_LEN):
    print("Tokenizando pares de frases...")
    src_ids, tgt_ids = tokenize_pairs(pairs, tokenizer, max_len)

    dataset = TranslationDataset(src_ids, tgt_ids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"DataLoader pronto: {len(dataset)} amostras, "
          f"batch_size={batch_size}, {len(loader)} batches/epoch")
    return loader, tokenizer.pad_token_id


if __name__ == "__main__":
    from scripts.dataset import load_translation_subset

    tokenizer = load_tokenizer()
    pairs = load_translation_subset(100)
    loader, pad_id = build_dataloader(pairs, tokenizer, batch_size=8)

    src_batch, tgt_in_batch, tgt_out_batch = next(iter(loader))
    print(f"\nFormato do batch:")
    print(f"  src shape      : {src_batch.shape}")
    print(f"  tgt_input shape: {tgt_in_batch.shape}")
    print(f"  tgt_target shape: {tgt_out_batch.shape}")
    print(f"  PAD token id   : {pad_id}")