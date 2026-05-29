from fastapi import APIRouter, HTTPException

from app.schemas.request_models import SearchRequest
from app.schemas.response_models import SearchResponse
from app.services.embedding_service import EmbeddingService
from app.services.postgres_vector_repository import (
    PostgresVectorRepository
)

router = APIRouter()

embedding_service = EmbeddingService()
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
