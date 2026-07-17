import sqlite3
import json
from pathlib import Path
from datetime import datetime


class Database:

    def __init__(self, db_path="data/discord.db"):

        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys = ON;")

        self.create_tables()

    ######################################################################

    def create_tables(self):

        self.conn.executescript("""

        CREATE TABLE IF NOT EXISTS messages(

            message_id          INTEGER PRIMARY KEY,

            channel_id          INTEGER,

            author_id           INTEGER,
            author_name         TEXT,
            display_name        TEXT,

            content             TEXT,

            created_at          TEXT,
            edited_at           TEXT,

            reply_to            INTEGER,

            jump_url            TEXT,

            has_attachment      INTEGER DEFAULT 0,

            json_blob           TEXT
        );

        CREATE TABLE IF NOT EXISTS attachments(

            id                  INTEGER PRIMARY KEY AUTOINCREMENT,

            message_id          INTEGER,

            filename            TEXT,
            url                 TEXT,

            FOREIGN KEY(message_id)
                REFERENCES messages(message_id)
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS embeddings(

            message_id          INTEGER PRIMARY KEY,

            faiss_id            INTEGER UNIQUE,

            embedding_model     TEXT,

            embedded_at         TEXT,

            FOREIGN KEY(message_id)
                REFERENCES messages(message_id)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_channel
        ON messages(channel_id);

        CREATE INDEX IF NOT EXISTS idx_author
        ON messages(author_id);

        CREATE INDEX IF NOT EXISTS idx_created
        ON messages(created_at);

        """)

        self.conn.commit()

    ######################################################################

    def import_jsonl(self, filename, channel_id):

        imported = 0

        with open(filename, "r", encoding="utf8") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                msg = json.loads(line)

                author = msg["author"]

                self.conn.execute("""
                INSERT OR REPLACE INTO messages
                (
                    message_id,
                    channel_id,
                    author_id,
                    author_name,
                    display_name,
                    content,
                    created_at,
                    edited_at,
                    reply_to,
                    jump_url,
                    has_attachment,
                    json_blob
                )
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    msg["id"],
                    channel_id,
                    author["id"],
                    author["name"],
                    author["display_name"],
                    msg.get("content", ""),
                    msg.get("created_at"),
                    msg.get("edited_at"),
                    msg.get("reply_to"),
                    msg.get("jump_url"),
                    int(len(msg.get("attachments", [])) > 0),
                    json.dumps(msg)
                ))

                for attachment in msg.get("attachments", []):

                    self.conn.execute("""
                    INSERT INTO attachments
                    (
                        message_id,
                        filename,
                        url
                    )
                    VALUES (?,?,?)
                    """,
                    (
                        msg["id"],
                        attachment.get("filename"),
                        attachment.get("url")
                    ))

                imported += 1

        self.conn.commit()

        print(f"Imported {imported:,} messages.")

    ######################################################################

    def get_message(self, message_id):

        cur = self.conn.execute("""
        SELECT *
        FROM messages
        WHERE message_id = ?
        """, (message_id,))

        return cur.fetchone()

    ######################################################################

    def get_unembedded_messages(self, limit=None):

        sql = """
        SELECT
            message_id,
            channel_id,
            author_id,
            author_name,
            display_name,
            content,
            created_at
        FROM messages
        WHERE message_id NOT IN
        (
            SELECT message_id
            FROM embeddings
        )
        AND TRIM(content) != ''
        ORDER BY created_at
        """

        if limit is not None:
            sql += " LIMIT ?"
            cur = self.conn.execute(sql, (limit,))
        else:
            cur = self.conn.execute(sql)

        return cur.fetchall()

    ######################################################################

    def add_embedding(self, message_id, faiss_id, model):

        self.conn.execute("""
        INSERT OR REPLACE INTO embeddings
        (
            message_id,
            faiss_id,
            embedding_model,
            embedded_at
        )
        VALUES (?,?,?,?)
        """,
        (
            message_id,
            faiss_id,
            model,
            datetime.utcnow().isoformat()
        ))

        self.conn.commit()

    ######################################################################

    def close(self):

        self.conn.close()

    ######################################################################

    def get_messages_by_faiss_ids(self, faiss_ids):

        # Convert NumPy array to a normal Python list and remove invalid IDs
        faiss_ids = [int(i) for i in faiss_ids if i >= 0]

        if len(faiss_ids) == 0:
            return []

        placeholders = ",".join(["?"] * len(faiss_ids))

        sql = f"""
        SELECT
            m.*,
            e.faiss_id
        FROM messages m
        JOIN embeddings e
            ON m.message_id = e.message_id
        WHERE e.faiss_id IN ({placeholders})
        """

        cur = self.conn.execute(sql, faiss_ids)

        rows = cur.fetchall()

        lookup = {row["faiss_id"]: row for row in rows}

        # Return rows in the same order as FAISS returned them
        return [lookup[i] for i in faiss_ids if i in lookup]