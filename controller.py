from fastapi import APIRouter, HTTPException, Query, FastAPI, Depends
from fastapi.responses import JSONResponse

from config.settings import MODELO_EMBEDDINGS, CHAVE_API_GEMINI
from schemas.request_models import SearchRequest
from schemas.response_models import SearchResponse
from services.embedding_service import EmbeddingService
from services.postgres_vector_repository import PostgresVectorRepository
from services.deputados_temas_service import DeputadoTemaService
from typing import Optional

router = APIRouter()

# Inicialização dos serviços
embedding_service = EmbeddingService(
    nome_modelo=MODELO_EMBEDDINGS,
    chave_api=CHAVE_API_GEMINI
)
repository = PostgresVectorRepository()
deputado_tema_service = DeputadoTemaService(csv_path="transcricoes_final.csv")

def get_deputado_tema_service():
    return deputado_tema_service

# --- MÉTODO POST ---
@router.post(
    "/search",
    response_model=SearchResponse
)
async def search_documents(request: SearchRequest):
    try:
        # Gera o embedding a partir do texto enviado no body JSON
        embedding = embedding_service.generate_embedding([request.query])[0]

        results = repository.search_similar_documents(
            embedding=embedding,
            top_k=request.top_k
        )

        return {"results": results}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# --- MÉTODO GET ---
@router.get(
    "/search",
    response_model=SearchResponse,
    response_class=JSONResponse
)
async def search_documents_get(
    query: str = Query(..., description="Texto a ser buscado"),
    top_k: int = Query(10, description="Quantidade de resultados")
):
    try:
        # Gera o embedding a partir do texto enviado na URL
        embedding = embedding_service.generate_embedding([query])[0]

        results = repository.search_similar_documents(
            embedding=embedding,
            top_k=top_k
        )

        return {"results": results}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
@router.get("/discursos/buscar")
def buscar(
    id_deputado: Optional[int] = None,
    nome_deputado: Optional[str] = None,
    tema: Optional[str] = None,
    service: DeputadoTemaService = Depends(get_deputado_tema_service),
):
    """Controller: Apenas recebe os parâmetros e repassa para a camada de Service."""

    # Chama a lógica de negócio isolada
    resultados = service.buscar_discursos(
        id_deputado=id_deputado, nome_deputado=nome_deputado, tema=tema
    )

    # Se a base estiver vazia por algum erro de carregamento no service
    if service.df.empty:
        raise HTTPException(
            status_code=500, detail="Base de dados indisponível."
        )

    return resultados