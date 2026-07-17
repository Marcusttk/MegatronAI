import faiss
import numpy as np
from pathlib import Path


class FaissIndex:

    def __init__(self,
                 dimension=768,
                 filename="data/embeddings.faiss"):

        self.dimension = dimension
        self.filename = Path(filename)

        self.filename.parent.mkdir(parents=True, exist_ok=True)

        if self.filename.exists():

            print("Loading existing FAISS index...")

            self.index = faiss.read_index(str(self.filename))

        else:

            print("Creating new FAISS index...")

            self.index = faiss.IndexFlatIP(dimension)

    #########################################################

    def add(self, embeddings):

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

    #########################################################

    def search(self, embedding, k=10):

        embedding = np.asarray(
            [embedding],
            dtype=np.float32
        )

        faiss.normalize_L2(embedding)

        scores, ids = self.index.search(
            embedding,
            k
        )

        return ids[0], scores[0]

    #########################################################

    def save(self):

        faiss.write_index(
            self.index,
            str(self.filename)
        )

    #########################################################

    @property
    def size(self):

        return self.index.ntotal