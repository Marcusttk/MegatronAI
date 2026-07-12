from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
import faiss
import numpy as np
import pickle

# ==========================
# SETTINGS
# ==========================

# Folder containing your downloaded Discord txt files
TXT_FOLDER = "discord_channels"

# Embedding model
EMBED_MODEL = "nomic-embed-text"

# Output files
FAISS_INDEX = "discord.index"
CHUNKS_FILE = "discord_chunks.pkl"

# Chunk settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ==========================
# LOAD TXT FILES
# ==========================

documents = []

txt_files = list(Path(TXT_FOLDER).glob("*.txt"))

if len(txt_files) == 0:
    raise Exception(f"No txt files found inside '{TXT_FOLDER}'")

for file in txt_files:
    print(f"Reading {file.name}")

    text = file.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    documents.append(
        {
            "source": file.name,
            "text": text
        }
    )

# ==========================
# SPLIT INTO CHUNKS
# ==========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunks = []

for doc in documents:

    split_text = splitter.split_text(doc["text"])

    for chunk in split_text:

        chunks.append(
            {
                "source": doc["source"],
                "text": chunk
            }
        )

print(f"\nCreated {len(chunks)} chunks.")

# ==========================
# CREATE EMBEDDINGS
# ==========================

texts = [c["text"] for c in chunks]

print("Generating embeddings...")

response = ollama.embed(
    model=EMBED_MODEL,
    input=texts
)

embeddings = response["embeddings"]

print(f"Generated {len(embeddings)} embeddings.")

# ==========================
# BUILD FAISS INDEX
# ==========================

dimension = len(embeddings[0])

index = faiss.IndexFlatL2(dimension)

embedding_array = np.array(
    embeddings,
    dtype=np.float32
)

index.add(embedding_array)

# ==========================
# SAVE EVERYTHING
# ==========================

faiss.write_index(index, FAISS_INDEX)

with open(CHUNKS_FILE, "wb") as f:
    pickle.dump(chunks, f)

print("\nFinished!")
print(f"FAISS index saved to: {FAISS_INDEX}")
print(f"Chunks saved to: {CHUNKS_FILE}")
print(f"Total chunks: {len(chunks)}")