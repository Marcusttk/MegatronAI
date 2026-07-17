from database import Database
from embedder import Embedder
from faiss_index import FaissIndex


class SearchEngine:

    def __init__(self):

        self.db = Database()

        self.embedder = Embedder()

        self.faiss = FaissIndex()

    ##############################################################

    def search(self, question, k=10):

        query_embedding = self.embedder.embed(question)

        faiss_ids, scores = self.faiss.search(
            query_embedding,
            k
        )

        results = self.db.get_messages_by_faiss_ids(faiss_ids)

        output = []

        for row, score in zip(results, scores):

            output.append({
                "score": float(score),
                "message_id": row["message_id"],
                "channel_id": row["channel_id"],
                "author": row["display_name"],
                "content": row["content"]
            })

        return output