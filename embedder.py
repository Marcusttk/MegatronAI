from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import ollama


DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_MAX_CHARACTERS = 8_000


class Embedder:
    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        max_characters: int = DEFAULT_MAX_CHARACTERS,
        retries: int = 4,
    ) -> None:
        self.model = model
        self.max_characters = max_characters
        self.retries = retries
        self.dimension: int | None = None

    def _clean_text(
        self,
        text: str,
    ) -> str:
        cleaned = str(text).replace("\x00", " ").strip()

        if len(cleaned) > self.max_characters:
            cleaned = cleaned[: self.max_characters]

        return cleaned

    def format_document(
        self,
        message: dict,
    ) -> str:
        author = (
            message.get("author_display_name")
            or message.get("author_name")
            or "Unknown"
        )

        channel = message.get("channel_name") or "Unknown"
        content = self._clean_text(message.get("content", ""))

        return (
            "search_document: "
            f"Author: {author}\n"
            f"Channel: {channel}\n"
            f"Message: {content}"
        )

    def format_query(
        self,
        query: str,
    ) -> str:
        return f"search_query: {self._clean_text(query)}"

    def embed_documents(
        self,
        messages: Sequence[dict],
    ) -> np.ndarray:
        texts = [
            self.format_document(message)
            for message in messages
        ]

        return self.embed_texts(texts)

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        vectors = self.embed_texts(
            [self.format_query(query)]
        )

        return vectors[0]

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        if not texts:
            raise ValueError("No texts were supplied for embedding.")

        last_error: Exception | None = None

        for attempt in range(1, self.retries + 1):
            try:
                response = ollama.embed(
                    model=self.model,
                    input=list(texts),
                )

                embeddings = response["embeddings"]

                vectors = np.asarray(
                    embeddings,
                    dtype=np.float32,
                )

                if vectors.ndim != 2:
                    raise ValueError(
                        f"Expected a 2D embedding array, got {vectors.shape}"
                    )

                if vectors.shape[0] != len(texts):
                    raise ValueError(
                        "Ollama returned a different number of embeddings "
                        "than the number of input texts."
                    )

                self._normalise(vectors)

                if self.dimension is None:
                    self.dimension = int(vectors.shape[1])
                elif vectors.shape[1] != self.dimension:
                    raise ValueError(
                        "Embedding dimension changed unexpectedly. "
                        f"Expected {self.dimension}, got {vectors.shape[1]}."
                    )

                return vectors

            except Exception as error:
                last_error = error

                if attempt >= self.retries:
                    break

                wait_seconds = 2 ** attempt

                print(
                    f"\nEmbedding attempt {attempt} failed: {error}"
                )
                print(
                    f"Retrying in {wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)

        raise RuntimeError(
            f"Embedding failed after {self.retries} attempts: {last_error}"
        )

    @staticmethod
    def _normalise(
        vectors: np.ndarray,
    ) -> None:
        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True,
        )

        norms[norms == 0] = 1.0
        vectors /= norms