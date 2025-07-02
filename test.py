import kaldiio
import numpy as np

scp_path = "embeddings/segments.scp"

# Correct way: returns a dictionary-like object
embeddings = kaldiio.load_scp(scp_path)

# Now iterate properly
for key in embeddings:
    emb = embeddings[key]
    print(f"{key}: shape={emb.shape}, mean={np.mean(emb):.4f}, norm={np.linalg.norm(emb):.4f}")
