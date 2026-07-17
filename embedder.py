from ollama import Client

client = Client(host="http://localhost:11434")


class Embedder:

    def __init__(self, model="nomic-embed-text"):

        self.model = model

    ###############################################################

    def embed(self, text):

        response = client.embed(
            model=self.model,
            input=text
        )

        return response.embeddings[0]

    ###############################################################

    def embed_batch(self, texts):

        response = client.embed(
            model=self.model,
            input=texts
        )

        return response.embeddings