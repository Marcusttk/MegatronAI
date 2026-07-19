from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

from tqdm import tqdm

from database import Database
from embedder import Embedder
from faiss_index import FaissIndex


DEFAULT_INPUT_DIRECTORY = Path(r"E:\discord_server_txt")
DEFAULT_DATABASE_PATH = Path("minder_msg/discord.db")
DEFAULT_INDEX_PATH = Path("minder_msg/embeddings.faiss")

IMPORT_BATCH_SIZE = 5_000
EMBEDDING_BATCH_SIZE = 128
SAVE_EVERY_BATCHES = 25


def safe_int(
    value: Any,
) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def clean_text(
    value: Any,
) -> str:
    if value is None:
        return ""

    return str(value).replace("\x00", " ").strip()


def parse_jump_url(
    jump_url: str,
) -> tuple[int | None, int | None]:
    if not jump_url:
        return None, None

    try:
        parsed = urlparse(jump_url)

        parts = [
            part
            for part in parsed.path.split("/")
            if part
        ]

        # /channels/guild_id/channel_id/message_id
        if len(parts) < 4 or parts[0] != "channels":
            return None, None

        guild_id = safe_int(parts[1])
        channel_id = safe_int(parts[2])

        return guild_id, channel_id

    except Exception:
        return None, None


def channel_name_from_path(
    jsonl_path: Path,
    input_directory: Path,
) -> str:
    relative = jsonl_path.relative_to(input_directory)
    return relative.with_suffix("").as_posix()


def convert_message(
    raw_message: dict[str, Any],
    jsonl_path: Path,
    input_directory: Path,
) -> dict[str, Any] | None:
    message_id = safe_int(raw_message.get("id"))

    if message_id is None:
        return None

    author = raw_message.get("author")

    if not isinstance(author, dict):
        author = {}

    jump_url = clean_text(raw_message.get("jump_url"))
    guild_id, channel_id = parse_jump_url(jump_url)

    return {
        "id": message_id,
        "guild_id": guild_id,
        "channel_id": channel_id,
        "channel_name": channel_name_from_path(
            jsonl_path,
            input_directory,
        ),
        "source_file": str(
            jsonl_path.relative_to(input_directory)
        ),
        "author_id": safe_int(author.get("id")),
        "author_name": clean_text(author.get("name")) or None,
        "author_display_name": (
            clean_text(author.get("display_name")) or None
        ),
        "author_bot": bool(author.get("bot", False)),
        "content": clean_text(raw_message.get("content")),
        "created_at": (
            clean_text(raw_message.get("created_at")) or None
        ),
        "edited_at": (
            clean_text(raw_message.get("edited_at")) or None
        ),
        "reply_to": safe_int(raw_message.get("reply_to")),
        "jump_url": jump_url or None,
    }


def iter_jsonl_messages(
    jsonl_path: Path,
    input_directory: Path,
) -> Iterator[dict[str, Any]]:
    with jsonl_path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                raw_message = json.loads(stripped)

                if not isinstance(raw_message, dict):
                    print(
                        f"\nSkipping non-object JSON in "
                        f"{jsonl_path}, line {line_number}"
                    )
                    continue

                message = convert_message(
                    raw_message=raw_message,
                    jsonl_path=jsonl_path,
                    input_directory=input_directory,
                )

                if message is not None:
                    yield message

            except json.JSONDecodeError as error:
                print(
                    f"\nSkipping invalid JSON in "
                    f"{jsonl_path}, line {line_number}: {error}"
                )


def import_jsonl_files(
    database: Database,
    input_directory: Path,
) -> None:
    jsonl_files = sorted(
        path
        for path in input_directory.rglob("*.jsonl")
        if path.is_file()
    )

    if not jsonl_files:
        raise FileNotFoundError(
            f"No JSONL files were found in {input_directory.resolve()}"
        )

    print(f"Found {len(jsonl_files)} JSONL files.")

    total_inserted = 0

    for jsonl_path in tqdm(
        jsonl_files,
        desc="Importing channels",
        unit="file",
    ):
        batch: list[dict[str, Any]] = []

        for message in iter_jsonl_messages(
            jsonl_path=jsonl_path,
            input_directory=input_directory,
        ):
            batch.append(message)

            if len(batch) >= IMPORT_BATCH_SIZE:
                total_inserted += database.insert_messages(batch)
                batch.clear()

        if batch:
            total_inserted += database.insert_messages(batch)

    print(f"New messages inserted: {total_inserted:,}")
    print(f"Total database messages: {database.count_messages():,}")


def build_embeddings(
    database: Database,
    embedder: Embedder,
    faiss_index: FaissIndex,
    batch_size: int,
) -> None:
    total_embeddable = database.count_embeddable_messages()
    already_embedded = database.count_embedded_messages()

    print(f"Embeddable messages: {total_embeddable:,}")
    print(f"Already embedded: {already_embedded:,}")
    print(f"Vectors currently in FAISS: {faiss_index.total_vectors:,}")

    remaining = max(
        total_embeddable - already_embedded,
        0,
    )

    if remaining == 0:
        print("No messages require embeddings.")
        return

    progress = tqdm(
        total=remaining,
        desc="Embedding messages",
        unit="message",
        initial=0,
    )

    completed_batches = 0

    while True:
        messages = database.get_unembedded_messages(
            limit=batch_size
        )

        if not messages:
            break

        vectors = embedder.embed_documents(messages)

        message_ids = [
            int(message["id"])
            for message in messages
        ]

        faiss_index.add(
            vectors=vectors,
            message_ids=message_ids,
        )

        # Save FAISS before marking SQLite rows complete.
        # This reduces the risk of SQLite claiming vectors exist
        # when the FAISS file has not yet been written.
        completed_batches += 1

        if completed_batches % SAVE_EVERY_BATCHES == 0:
            faiss_index.save()

        database.mark_embedded(message_ids)

        progress.update(len(messages))

    progress.close()

    print("Saving final FAISS index...")
    faiss_index.save()

    database.set_metadata(
        "embedding_model",
        embedder.model,
    )

    database.set_metadata(
        "embedding_dimension",
        str(faiss_index.dimension),
    )

    print(
        f"Finished. FAISS now contains "
        f"{faiss_index.total_vectors:,} vectors."
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import Discord JSONL files once and build a FAISS index."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIRECTORY,
        help="Folder containing channel-separated JSONL files.",
    )

    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help="SQLite database path.",
    )

    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="FAISS index path.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=EMBEDDING_BATCH_SIZE,
        help="Number of messages sent to Ollama per embedding request.",
    )

    parser.add_argument(
        "--skip-import",
        action="store_true",
        help="Skip JSONL import and only continue embedding.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    input_directory = arguments.input.resolve()

    with Database(arguments.database) as database:
        if not arguments.skip_import:
            import_jsonl_files(
                database=database,
                input_directory=input_directory,
            )

        embedder = Embedder(
            model="nomic-embed-text"
        )

        faiss_index = FaissIndex(
            index_path=arguments.index,
        )

        build_embeddings(
            database=database,
            embedder=embedder,
            faiss_index=faiss_index,
            batch_size=arguments.batch_size,
        )


if __name__ == "__main__":
    main()