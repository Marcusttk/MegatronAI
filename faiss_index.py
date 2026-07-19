from __future__ import annotations

from pathlib import Path
from typing import Sequence

import faiss
import numpy as np


DEFAULT_INDEX_PATH = Path("data/embeddings.faiss")


class FaissIndex:
    def __init__(
        self,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        dimension: int | None = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists():
            print("Loading existing FAISS index...")

            self.index = faiss.read_index(
                str(self.index_path)
            )

            self.dimension = int(self.index.d)

        else:
            if dimension is None:
                self.index = None
                self.dimension = None
            else:
                self.dimension = int(dimension)
                self.index = self._create_index(self.dimension)

    @staticmethod
    def _create_index(
        dimension: int,
    ) -> faiss.Index:
        flat_index = faiss.IndexFlatIP(dimension)
        return faiss.IndexIDMap2(flat_index)

    def ensure_created(
        self,
        dimension: int,
    ) -> None:
        if self.index is None:
            self.dimension = int(dimension)
            self.index = self._create_index(self.dimension)

        elif self.dimension != dimension:
            raise ValueError(
                "FAISS dimension does not match embedding dimension. "
                f"FAISS={self.dimension}, embedding={dimension}"
            )

    def add(
        self,
        vectors: np.ndarray,
        message_ids: Sequence[int],
    ) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        if vectors.ndim != 2:
            raise ValueError(
                f"Expected two-dimensional vectors, got {vectors.shape}"
            )

        if len(vectors) != len(message_ids):
            raise ValueError(
                "The vector count and message ID count do not match."
            )

        self.ensure_created(vectors.shape[1])

        ids = np.asarray(
            message_ids,
            dtype=np.int64,
        )

        self.index.add_with_ids(
            np.ascontiguousarray(vectors),
            ids,
        )

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("The FAISS index has not been created.")

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        scores, message_ids = self.index.search(
            np.ascontiguousarray(query_vector),
            k,
        )

        return scores[0], message_ids[0]

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("There is no FAISS index to save.")

        temporary_path = self.index_path.with_suffix(
            self.index_path.suffix + ".tmp"
        )

        faiss.write_index(
            self.index,
            str(temporary_path),
        )

        temporary_path.replace(self.index_path)

    @property
    def total_vectors(self) -> int:
        if self.index is None:
            return 0

        return int(self.index.ntotal)