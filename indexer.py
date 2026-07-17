from database import Database
from embedder import Embedder
from faiss_index import FaissIndex

BATCH_SIZE = 128

db = Database()
embedder = Embedder()
faiss_db = FaissIndex()

total_added = 0

while True:

    messages = db.get_unembedded_messages(BATCH_SIZE)

    if len(messages) == 0:
        break

    texts = []
    message_ids = []

    for row in messages:

        # Build the text that gets embedded
        text = f"""
Author: {row['display_name']}

Message:
{row['content']}
""".strip()

        texts.append(text)
        message_ids.append(row["message_id"])

    print(f"Embedding {len(texts)} messages...")

    vectors = embedder.embed_batch(texts)

    start_id = faiss_db.size

    faiss_db.add(vectors)

    for i, message_id in enumerate(message_ids):

        db.add_embedding(
            message_id=message_id,
            faiss_id=start_id + i,
            model="nomic-embed-text"
        )

    faiss_db.save()

    total_added += len(vectors)

    print(f"Total indexed: {faiss_db.size}")

print()
print("=" * 50)
print("Indexing complete.")
print(f"New embeddings added: {total_added}")
print(f"Total vectors in FAISS: {faiss_db.size}")

db.close()