import json
import ollama


MODEL = "nomic-embed-text"
JSONL_FILE = "intros.jsonl"
OUTPUT_FILE = "embeddings.json"


def build_text(message):
    """
    Convert a Discord message into searchable text.
    """

    content = message.get("content", "").strip()

    if not content:
        return None

    author = message["author"]["display_name"]

    return f"Author: {author}\nMessage: {content}"


def embed_jsonl_batch(jsonl_path, batch_size=64):
    """
    Read a JSONL file and generate embeddings in batches.

    Yields:
        embeddings
        metadata
    """

    batch_text = []
    batch_metadata = []

    with open(jsonl_path, "r", encoding="utf-8") as f:

        for line in f:

            message = json.loads(line)

            text = build_text(message)

            if text is None:
                continue

            batch_text.append(text)

            batch_metadata.append(message)

            if len(batch_text) >= batch_size:

                response = ollama.embed(
                    model=MODEL,
                    input=batch_text
                )

                yield response["embeddings"], batch_metadata

                batch_text = []
                batch_metadata = []

        if batch_text:

            response = ollama.embed(
                model=MODEL,
                input=batch_text
            )

            yield response["embeddings"], batch_metadata


def main():

    all_embeddings = []

    total = 0

    for embeddings, messages in embed_jsonl_batch(JSONL_FILE):

        for embedding, message in zip(embeddings, messages):

            all_embeddings.append({
                "id": message["id"],
                "content": message["content"],
                "author": message["author"]["display_name"],
                "created_at": message["created_at"],
                "embedding": embedding
            })

            total += 1

        print(f"Embedded {total} messages...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            all_embeddings,
            f,
            ensure_ascii=False,
            indent=4
        )

    print(f"\nFinished!")
    print(f"Messages embedded: {total}")
    print(f"Saved to: {OUTPUT_FILE}")

    if total:
        print(f"Embedding dimension: {len(all_embeddings[0]['embedding'])}")


if __name__ == "__main__":
    main()