import json
import ollama


def build_text(message):
    """
    Convert a Discord message into searchable text.
    """

    author = message["author"]["display_name"]
    content = message["content"].strip()

    if not content:
        return None

    return f"Author: {author}\nMessage: {content}"


def generate_embedding(text, model="nomic-embed-text"):
    """
    Generate a single embedding using Ollama.
    """

    response = ollama.embed(
        model=model,
        input=text
    )

    return response["embeddings"][0]


def embed_jsonl(jsonl_path, model="nomic-embed-text"):
    """
    Read a JSONL file and return embeddings + metadata.
    """

    embeddings = []
    metadata = []

    with open(jsonl_path, "r", encoding="utf-8") as f:

        for line in f:

            message = json.loads(line)

            text = build_text(message)

            if text is None:
                continue

            embedding = generate_embedding(text, model)

            embeddings.append(embedding)

            metadata.append({
                "id": message["id"],
                "channel": jsonl_path,
                "content": message["content"],
                "author": message["author"]["display_name"],
                "created_at": message["created_at"]
            })

    return embeddings, metadata


def embed_jsonl_batch(jsonl_path, batch_size=128, model="nomic-embed-text"):
    """
    Reads a JSONL file and yields batches of embeddings and metadata.

    Returns:
        embeddings : list[list[float]]
        metadata   : list[dict]
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

            batch_metadata.append({
                "id": message["id"],
                "content": message["content"],
                "author": message["author"]["display_name"],
                "created_at": message["created_at"]
            })

            if len(batch_text) == batch_size:

                response = ollama.embed(
                    model=model,
                    input=batch_text
                )

                yield response["embeddings"], batch_metadata

                batch_text = []
                batch_metadata = []

        # Embed any remaining messages
        if batch_text:

            response = ollama.embed(
                model=model,
                input=batch_text
            )

            yield response["embeddings"], batch_metadata


if __name__ == "__main__":
    paths = [
        "E:/discord_server_txt/",  # Desktop
        "/media/pi/Transcend/discord_server_txt/",  # Raspberry Pi
    ]
    jsonl_test = "./intros.jsonl"
    embeddings, metadata = embed_jsonl(jsonl_test)