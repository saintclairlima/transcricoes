from fastapi import APIRouter, HTTPException

from transcricoes.config.settings import MODELO_EMBEDDINGS
from transcricoes.config.settings import CHAVE_API_GEMINI

from transcricoes.schemas.request_models import SearchRequest
from transcricoes.schemas.response_models import SearchResponse
from transcricoes.services.embedding_service import EmbeddingService
from transcricoes.services.postgres_vector_repository import (
    PostgresVectorRepository
)

router = APIRouter()

embedding_service = EmbeddingService(
    nome_modelo = MODELO_EMBEDDINGS,
    chave_api = CHAVE_API_GEMINI
)
repository = PostgresVectorRepository()


@router.post(
    "/search",
    response_model=SearchResponse
)
async def search_documents(request: SearchRequest):

    try:

        embedding = embedding_service.generate_embedding(
            request.query
        )

        results = repository.search_similar_documents(
            embedding=embedding,
            top_k=request.top_k
        )

        return {
            "results": results
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
