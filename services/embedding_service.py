from openai import OpenAI
from app.config.settings import EMBEDDING_MODEL

class EmbeddingService:

    def __init__(self):
        self.client = OpenAI()

    def generate_embedding(self, text: str):

        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )

        return response.data[0].embedding
