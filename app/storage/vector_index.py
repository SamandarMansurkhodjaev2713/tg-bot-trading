import faiss
import numpy as np

class NewsIndex:
    def __init__(self, dim: int = 384):
        self.index = faiss.IndexFlatIP(dim)
        self.ids = []

    def add(self, embeddings: np.ndarray, ids: list[str]):
        self.index.add(embeddings.astype(np.float32))
        self.ids.extend(ids)

    def search(self, query: np.ndarray, k: int = 10):
        d, I = self.index.search(query.astype(np.float32), k)
        return [(self.ids[i], float(dj)) for dj, i in zip(d[0], I[0])]