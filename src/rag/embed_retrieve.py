import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from sklearn.preprocessing import normalize

from src.config import EMBED_MODEL


class EmbeddingManager:

    def __init__(self, model_name: str = EMBED_MODEL):

        self.model_name = model_name
        self.model = None

        self._load_model()
    def _load_model(self):
        try:
            print(f"Loading embedding model:{self.model_name}")
            self.model=SentenceTransformer(self.model_name)
            print(f"Model loaded successfully.Embedding dimension:{self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loaging model{self.model_name}:{e}")
            raise

    def generate_embedding(self,texts:List[str])->np.array:
        if not self.model:
            raise ValueError("Model not loaded")
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings=self.model.encode(texts,show_progress_bar=True)
        print(f"Generated embeddins with shape:{embeddings.shape}")
        return embeddings
    


class Retriever:

    def __init__(self, store):
        self.store = store

    def search(self, query_embedding: np.ndarray, top_k: int = 5):

        if self.store.index.ntotal == 0:
            raise ValueError("No embeddings stored")

        query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)

        # normalize query too
        query_embedding = normalize(query_embedding)

        # 🔥 FAISS search
        scores, indices = self.store.index.search(query_embedding, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            if idx >= len(self.store.documents):
                continue

            results.append({
                "text": self.store.documents[idx]["text"],
                "source": self.store.documents[idx]["source"],
                "score": float(score)
            })

        return results