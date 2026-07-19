from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_DATABASE_PATH = Path("data/discord.db")


class Database:
    def __init__(
        self,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
    ) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(
            self.database_path,
            timeout=60,
        )

        self.connection.row_factory = sqlite3.Row

        self._configure()
        self._create_tables()

    def _configure(self) -> None:
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self.connection.execute("PRAGMA temp_store = MEMORY")
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 60000")

    def _create_tables(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id                  INTEGER PRIMARY KEY,
                guild_id            INTEGER,
                channel_id          INTEGER,
                channel_name        TEXT NOT NULL,
                source_file         TEXT NOT NULL,

                author_id           INTEGER,
                author_name         TEXT,
                author_display_name TEXT,
                author_bot          INTEGER NOT NULL DEFAULT 0,

                content             TEXT NOT NULL,
                created_at          TEXT,
                edited_at           TEXT,
                reply_to            INTEGER,
                jump_url            TEXT,

                embedded            INTEGER NOT NULL DEFAULT 0,
                imported_at         TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_messages_channel
                ON messages(channel_id);

            CREATE INDEX IF NOT EXISTS idx_messages_author
                ON messages(author_id);

            CREATE INDEX IF NOT EXISTS idx_messages_created
                ON messages(created_at);

            CREATE INDEX IF NOT EXISTS idx_messages_embedded
                ON messages(embedded);

            CREATE INDEX IF NOT EXISTS idx_messages_channel_name
                ON messages(channel_name);


            CREATE TABLE IF NOT EXISTS build_metadata (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )

        self.connection.commit()

    def insert_messages(
        self,
        messages: Sequence[dict[str, Any]],
    ) -> int:
        if not messages:
            return 0

        rows = [
            (
                message["id"],
                message.get("guild_id"),
                message.get("channel_id"),
                message["channel_name"],
                message["source_file"],
                message.get("author_id"),
                message.get("author_name"),
                message.get("author_display_name"),
                int(bool(message.get("author_bot", False))),
                message.get("content", ""),
                message.get("created_at"),
                message.get("edited_at"),
                message.get("reply_to"),
                message.get("jump_url"),
            )
            for message in messages
        ]

        before = self.connection.total_changes

        self.connection.executemany(
            """
            INSERT OR IGNORE INTO messages (
                id,
                guild_id,
                channel_id,
                channel_name,
                source_file,

                author_id,
                author_name,
                author_display_name,
                author_bot,

                content,
                created_at,
                edited_at,
                reply_to,
                jump_url
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        self.connection.commit()

        return self.connection.total_changes - before

    def insert_live_message(
        self,
        message: dict[str, Any],
    ) -> bool:
        inserted = self.insert_messages([message])
        return inserted == 1

    def get_unembedded_messages(
        self,
        limit: int,
    ) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT
                id,
                guild_id,
                channel_id,
                channel_name,
                author_id,
                author_name,
                author_display_name,
                content,
                created_at,
                edited_at,
                reply_to,
                jump_url
            FROM messages
            WHERE embedded = 0
              AND TRIM(content) != ''
              AND author_bot = 0
            ORDER BY id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [dict(row) for row in rows]

    def mark_embedded(
        self,
        message_ids: Iterable[int],
    ) -> None:
        ids = [(int(message_id),) for message_id in message_ids]

        if not ids:
            return

        self.connection.executemany(
            """
            UPDATE messages
            SET embedded = 1
            WHERE id = ?
            """,
            ids,
        )

        self.connection.commit()

    def reset_embedding_state(self) -> None:
        self.connection.execute(
            """
            UPDATE messages
            SET embedded = 0
            """
        )

        self.connection.commit()

    def count_messages(self) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM messages"
        ).fetchone()

        return int(row["count"])

    def count_embeddable_messages(self) -> int:
        row = self.connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM messages
            WHERE TRIM(content) != ''
              AND author_bot = 0
            """
        ).fetchone()

        return int(row["count"])

    def count_embedded_messages(self) -> int:
        row = self.connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM messages
            WHERE embedded = 1
            """
        ).fetchone()

        return int(row["count"])

    def get_messages_by_ids(
        self,
        message_ids: Sequence[int],
    ) -> list[dict[str, Any]]:
        if not message_ids:
            return []

        placeholders = ",".join("?" for _ in message_ids)

        rows = self.connection.execute(
            f"""
            SELECT
                id,
                guild_id,
                channel_id,
                channel_name,

                author_id,
                author_name,
                author_display_name,

                content,
                created_at,
                edited_at,
                reply_to,
                jump_url
            FROM messages
            WHERE id IN ({placeholders})
            """,
            tuple(int(message_id) for message_id in message_ids),
        ).fetchall()

        messages_by_id = {
            int(row["id"]): dict(row)
            for row in rows
        }

        return [
            messages_by_id[message_id]
            for message_id in message_ids
            if message_id in messages_by_id
        ]

    def set_metadata(
        self,
        key: str,
        value: str,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO build_metadata(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value
            """,
            (key, value),
        )

        self.connection.commit()

    def get_metadata(
        self,
        key: str,
    ) -> str | None:
        row = self.connection.execute(
            """
            SELECT value
            FROM build_metadata
            WHERE key = ?
            """,
            (key,),
        ).fetchone()

        if row is None:
            return None

        return str(row["value"])

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()