import json
import kaldiio
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ---- Step 0: Load segment mapping from JSON ----


def load_segment_mapping(mapping_path):
    with open(mapping_path, "r") as f:
        segment_mapping = json.load(f)  # list of [start, end, segment_name]
    # Convert inner lists to tuples for convenience
    return [(float(s), float(e), name) for s, e, name in segment_mapping]

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

# ---- Step 2: Spectral clustering with cosine similarity ----
def spectral_cluster(embeddings, n_speakers=2):
    similarity = cosine_similarity(embeddings)
    clusterer = SpectralClustering(
        n_clusters=n_speakers,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42
    )
    labels = clusterer.fit_predict(similarity)
    return labels

# ---- Merge adjacent/overlapping segments with the same speaker ----
def merge_segments(segments, labels):
    if not segments:
        return []

    merged = []
    prev_start, prev_end = segments[0]
    prev_label = labels[0]

    for i in range(1, len(segments)):
        start, end = segments[i]
        label = labels[i]

        # Merge if same speaker and segments overlap or touch
        if label == prev_label and start <= prev_end:
            prev_end = max(prev_end, end)
        else:
            merged.append((prev_start, prev_end, prev_label))
            prev_start, prev_end, prev_label = start, end, label

    merged.append((prev_start, prev_end, prev_label))
    return merged

# ---- Step 3: Run everything and output merged timeline ----
def run_clustering(ark_path, mapping_path, n_speakers=2):
    # Load segment mapping
    segment_mapping = load_segment_mapping(mapping_path)

    # Load embeddings and names
    embeddings, segment_names = load_embeddings(ark_path)

    # Map embeddings to segments by matching segment names (keys)
    name_to_segment = {name: (start, end) for start, end, name in segment_mapping}
    segments_ordered = [name_to_segment[name] for name in segment_names]

    # Cluster embeddings
    labels = spectral_cluster(embeddings, n_speakers)

    # Merge same-speaker adjacent segments
    merged_segments = merge_segments(segments_ordered, labels)

    # Print final speaker timeline
    print("\n=== Speaker Timeline ===")
    for start, end, label in merged_segments:
        print(f"{start:.2f}s - {end:.2f}s → Speaker {label}")
    print("========================")

    return merged_segments, labels

# Usage example:
run_clustering("segments.ark", "segment_mapping.json", n_speakers=2)
