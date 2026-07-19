from __future__ import annotations

from pathlib import Path
from typing import Any

from database import Database
from embedder import Embedder
from faiss_index import FaissIndex


DEFAULT_DATABASE_PATH = Path("minder_msg/discord.db")
DEFAULT_INDEX_PATH = Path("minder_msg/embeddings.faiss")


class SearchEngine:
    def __init__(
        self,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        embedding_model: str = "nomic-embed-text",
    ) -> None:
        self.database = Database(database_path)

        self.embedder = Embedder(
            model=embedding_model,
        )

        self.faiss_index = FaissIndex(
            index_path=index_path,
        )

        if self.faiss_index.total_vectors == 0:
            raise RuntimeError(
                "The FAISS index contains no vectors. "
                "Run build_database.py first."
            )

        stored_model = self.database.get_metadata(
            "embedding_model"
        )

        if stored_model and stored_model != embedding_model:
            raise RuntimeError(
                "Embedding model mismatch. "
                f"Index used {stored_model}, but search is using "
                f"{embedding_model}."
            )

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[dict[str, Any]]:
        query = query.strip()

        if not query:
            return []

        query_vector = self.embedder.embed_query(query)

        scores, ids = self.faiss_index.search(
            query_vector=query_vector,
            k=k,
        )

        valid_results = [
            (int(message_id), float(score))
            for score, message_id in zip(scores, ids)
            if int(message_id) != -1
        ]

        if not valid_results:
            return []

        ordered_ids = [
            message_id
            for message_id, _ in valid_results
        ]

        messages = self.database.get_messages_by_ids(
            ordered_ids
        )

        score_by_id = {
            message_id: score
            for message_id, score in valid_results
        }

        results: list[dict[str, Any]] = []

        for message in messages:
            message_id = int(message["id"])

            author = (
                message.get("author_display_name")
                or message.get("author_name")
                or "Unknown"
            )

            results.append(
                {
                    "id": message_id,
                    "score": score_by_id[message_id],
                    "author_id": message.get("author_id"),
                    "author": author,
                    "author_name": message.get("author_name"),
                    "display_name": message.get(
                        "author_display_name"
                    ),
                    "guild_id": message.get("guild_id"),
                    "channel_id": message.get("channel_id"),
                    "channel_name": message.get("channel_name"),
                    "content": message.get("content", ""),
                    "created_at": message.get("created_at"),
                    "edited_at": message.get("edited_at"),
                    "reply_to": message.get("reply_to"),
                    "jump_url": message.get("jump_url"),
                }
            )

        return results

    def close(self) -> None:
        self.database.close()

    def __enter__(self) -> "SearchEngine":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()