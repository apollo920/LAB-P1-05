"""
Tarefa 1 — Preparando o Dataset Real (Hugging Face)

Carrega o dataset bentrevett/multi30k (pares en-de) e seleciona
um subconjunto de 1000 frases para o treinamento.
"""

from datasets import load_dataset


def load_translation_subset(num_samples=1000):
    print("Carregando dataset multi30k do Hugging Face...")

    dataset = load_dataset("bentrevett/multi30k", trust_remote_code=True)
    train_split = dataset["train"]

    # Seleciona apenas as primeiras num_samples frases
    subset = train_split.select(range(num_samples))

    pairs = []
    for item in subset:
        pairs.append({
            "en": item["en"].strip(),
            "de": item["de"].strip(),
        })

    print(f"Dataset carregado: {len(pairs)} pares de frases (en -> de)")
    print(f"Exemplo: '{pairs[0]['en']}' -> '{pairs[0]['de']}'")

    return pairs


if __name__ == "__main__":
    pairs = load_translation_subset(1000)
    print(f"\nTotal de pares: {len(pairs)}")