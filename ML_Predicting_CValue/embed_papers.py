# embed_papers.py

import torch
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np

# ---------------------------
# Configuration
# ---------------------------
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "prospectis"
COLLECTION_NAME = "research_papers"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_FIELD = "scibert_embedding"

# ---------------------------
# Load SciBERT
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(DEVICE)
model.eval()

# ---------------------------
# Load MongoDB Papers
# ---------------------------
client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

# Only embed papers without existing embedding
cursor = collection.find({EMBED_FIELD: {"$exists": False}})
papers = list(cursor)

print(f"Embedding {len(papers)} papers...")

# ---------------------------
# Helper: BERT Embedding
# ---------------------------
def get_embedding(texts):
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**tokens)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

    return cls_embeddings.cpu().numpy()

# ---------------------------
# Batch Embedding Loop
# ---------------------------
for i in tqdm(range(0, len(papers), BATCH_SIZE)):
    batch = papers[i:i + BATCH_SIZE]
    texts = []

    for p in batch:
        title = p.get("title", "")
        abstract = p.get("abstract", "")
        combined = f"{title} {abstract}".strip()
        if not combined:
            combined = "[No Content]"
        texts.append(combined)

    embeddings = get_embedding(texts)

    for j, p in enumerate(batch):
        embedding_vector = embeddings[j].tolist()  # convert to native list
        collection.update_one(
            {"_id": p["_id"]},
            {"$set": {EMBED_FIELD: embedding_vector}}
        )

print("✅ Done embedding papers.")