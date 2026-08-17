import os
from typing import List, Optional, Sequence, Tuple
from fastembed import SparseTextEmbedding


class SpladeEmbeddingGenerator:
    """Generates learned sparse embeddings (SPLADE) with neural term expansion."""

    def __init__(
        self,
        model_name: str = "prithivida/Splade_PP_en_v1",
        batch_size: int = 32,
        threads: Optional[int] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.threads = threads or os.cpu_count()
        self._model: SparseTextEmbedding | None = None

    @property
    def model(self) -> SparseTextEmbedding:
        if self._model is None:
            self._model = SparseTextEmbedding(model_name=self.model_name, threads=self.threads)
        return self._model

    def embed_texts(self, texts: Sequence[str]) -> List[Tuple[List[int], List[float]]]:
        """Generates SPLADE sparse vectors (indices, values) for document chunks."""
        if not texts:
            return []
        embeddings = self.model.embed(texts, batch_size=self.batch_size)
        return [(emb.indices.tolist(), emb.values.tolist()) for emb in embeddings]

    def embed_query(self, query: str) -> Tuple[List[int], List[float]]:
        """Generates SPLADE sparse vector (indices, values) with query expansion."""
        emb_iter = self.model.embed([query], batch_size=1)
        emb = next(emb_iter)
        return emb.indices.tolist(), emb.values.tolist()

