from google import genai
from app.config.settings import EMBEDDING_MODEL

class EmbeddingService:

    def __init__(self, nome_modelo: str, chave_api: str, tamanho_lote: int=10):
        if not chave_api:
            raise ValueError(
                'A variável de ambiente GEMINI_API_KEY deve estar definida para usar o gerador de embeddings Gemini.'
            )
        self.nome_modelo = nome_modelo
        self.cliente = genai.Client(api_key=chave_api)
        self.tamanho_lote = tamanho_lote

    def generate_embedding(self, input: list[str]):
        """
        Gera embeddings fatiando o input para respeitar os limites de contexto.
        """
        embeddings = []
        total_documentos = len(input)

        for i in range(0, total_documentos, self.tamanho_lote):
            lote_atual = input[i : i + self.tamanho_lote]
            
            result = self.cliente.models.embed_content(
                model=self.nome_modelo,
                contents=lote_atual
            )
            
            for embedding in result.embeddings:
                embeddings.append(embedding.values)

        return embeddings
