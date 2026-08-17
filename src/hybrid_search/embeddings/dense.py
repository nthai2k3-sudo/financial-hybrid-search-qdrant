import os
from typing import List, Optional, Sequence
from fastembed import TextEmbedding


class DenseEmbeddingGenerator:
    """Generates dense vector embeddings using BAAI/bge-small-en-v1.5 via ONNX runtime."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 64,
        threads: Optional[int] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.threads = threads or os.cpu_count()
        self._model: TextEmbedding | None = None

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            self._model = TextEmbedding(model_name=self.model_name, threads=self.threads)
        return self._model

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Generates dense embeddings for a list of document passages."""
        if not texts:
            return []
        embeddings = self.model.embed(texts, batch_size=self.batch_size)
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """Generates dense embedding for a single search query."""
        query_text = f"Represent this sentence for searching relevant passages: {query}" if "bge" in self.model_name.lower() else query
        emb_iter = self.model.embed([query_text], batch_size=1)
        return next(emb_iter).tolist()

