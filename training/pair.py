import torch
import kaldiio
import itertools
import random
import glob
import os

def load_ark_embeddings(path):
    return [torch.tensor(vec, dtype=torch.float32) for key, vec in kaldiio.load_ark(path)]

def generate_balanced_pairs(ark_dir, total_pairs=15000, save_path="pairs.pt"):
    speaker_files = sorted(glob.glob(os.path.join(ark_dir, "*.ark")))
    speaker_embeddings = {os.path.basename(f): load_ark_embeddings(f) for f in speaker_files}

    positive_pairs = []
    negative_pairs = []

    # ---- Generate positive pairs (within each speaker) ----
    for __, embs in speaker_embeddings.items():
        pos = list(itertools.combinations(embs, 2))
        positive_pairs.extend([(a, b, torch.tensor(1.0)) for a, b in pos])

    # ---- Generate negative pairs (across different speakers) ----
    speaker_names = list(speaker_embeddings.keys())
    for i in range(len(speaker_names)):
        for j in range(i + 1, len(speaker_names)):
            embs1 = speaker_embeddings[speaker_names[i]]
            embs2 = speaker_embeddings[speaker_names[j]]
            neg = list(itertools.product(embs1, embs2))
            negative_pairs.extend([(a, b, torch.tensor(0.0)) for a, b in neg])

    # ---- Shuffle and trim to balance ----
    random.shuffle(positive_pairs)
    random.shuffle(negative_pairs)

    num_each = total_pairs // 2
    balanced_pos = positive_pairs[:num_each]
    balanced_neg = negative_pairs[:num_each]
    all_pairs = balanced_pos + balanced_neg
    random.shuffle(all_pairs)

    print(f"Total positive pairs: {len(balanced_pos)}")
    print(f"Total negative pairs: {len(balanced_neg)}")
    print(f"Total saved pairs: {len(all_pairs)}")

    torch.save(all_pairs, save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    generate_balanced_pairs("speakers/embeddings", total_pairs=15000)
