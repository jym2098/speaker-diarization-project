import numpy as np
from sklearn.cluster import KMeans
from kaldiio import load_scp

# Load embeddings
scp_path = "embeddings/segments.scp"
embeddings_dict = load_scp(scp_path)

# Convert to matrix and filter out any embeddings with NaNs
keys = []
vectors = []

for k, v in embeddings_dict.items():
    if not np.isnan(v).any():
        keys.append(k)
        vectors.append(v)
    else:
        print(f"⚠️ Skipping {k} due to NaN values.")

embeddings = np.stack(vectors)

# Run KMeans
n_clusters = 2  # or whatever you expect
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
labels = kmeans.labels_

# Print result
for i, key in enumerate(keys):
    print(f"{key}: Cluster {labels[i]}")
