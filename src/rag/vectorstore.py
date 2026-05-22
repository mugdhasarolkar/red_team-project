import faiss
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import normalize


class FAISSStore:

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim

        # cosine similarity = inner product on normalized vectors
        self.index = faiss.IndexFlatIP(embedding_dim)

        self.documents: List[Dict] = []

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Dict]):

        if len(embeddings) != len(metadatas):
            raise ValueError("Mismatch between embeddings and metadata")

        embeddings = np.array(embeddings).astype("float32")

        # 🔥 normalize for cosine similarity
        embeddings = normalize(embeddings)

        self.index.add(embeddings)

        self.documents = metadatas

        print(f"Stored {len(embeddings)} vectors in FAISS")