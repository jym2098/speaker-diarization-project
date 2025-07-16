import kaldiio
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize

# ---- Step 1: Load and normalize embeddings ----
def load_embeddings(ark_path):
    embeddings = []
    segment_names = []
    for key, vec in kaldiio.load_ark(ark_path):
        vec = np.array(vec)
        if np.isnan(vec).any():
            print(f"Skipping {key}: contains NaN")
            continue
        embeddings.append(vec)
        segment_names.append(key)

    if not embeddings:
        raise ValueError("No valid embeddings found.")

    embeddings = normalize(np.stack(embeddings), norm='l2')  # L2 normalize
    return embeddings, segment_names

# ---- Step 2: Agglomerative clustering with cosine distance ----
def agglomerative_cluster(embeddings, distance_threshold=0.4):
    distances = cosine_distances(embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(distances)
    return labels

# ---- Step 3: Run everything ----
def run_clustering(ark_path):
    embeddings, segment_names = load_embeddings(ark_path)
    labels = agglomerative_cluster(embeddings)

    for key, label in zip(segment_names, labels):
        print(f"{key} → Speaker {label}")

run_clustering("segments.ark")