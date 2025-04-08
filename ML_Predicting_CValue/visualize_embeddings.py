# visualize_embeddings.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pymongo import MongoClient
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# --------------- CONFIG ------------------
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "prospectis"
COLLECTION = "research_papers"
EMBED_FIELD = "scibert_embedding"
LABEL_FIELD = "categories"  # or something you define manually
REDUCE_METHOD = "pca"  # or "tsne"
MAX_POINTS = 300  # for performance
# -----------------------------------------

client = MongoClient(MONGO_URI)
papers = list(client[DB_NAME][COLLECTION].find(
    {EMBED_FIELD: {"$exists": True}},
    {EMBED_FIELD: 1, LABEL_FIELD: 1}
).limit(MAX_POINTS))

if not papers:
    print("No embeddings found! Are they saved under the right field?")
    exit()

# Load vectors + labels
vectors = np.array([p[EMBED_FIELD] for p in papers])
def get_label(paper):
    value = paper.get(LABEL_FIELD)
    if isinstance(value, list) and value:
        return value[0]
    elif isinstance(value, str):
        return value
    else:
        return "Unknown"

labels = [get_label(p) for p in papers]

# Reduce dimensions
if REDUCE_METHOD == "pca":
    reducer = PCA(n_components=2)
else:
    reducer = TSNE(n_components=2, perplexity=30, random_state=42)

reduced = reducer.fit_transform(vectors)

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="tab10", s=60)
plt.title(f"{REDUCE_METHOD.upper()} Projection of SciBERT Embeddings")
plt.legend(loc='best', fontsize="small", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()