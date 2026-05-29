from fastapi import APIRouter, HTTPException

from config.settings import MODELO_EMBEDDINGS
from config.settings import CHAVE_API_GEMINI

from schemas.request_models import SearchRequest
from schemas.response_models import SearchResponse
from services.embedding_service import EmbeddingService
from services.postgres_vector_repository import (
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

        # embedding = embedding_service.generate_embedding(
        #     request.query
        # )
        embedding = embedding_service.generate_embedding(
            [request.query]
        )[0]

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
