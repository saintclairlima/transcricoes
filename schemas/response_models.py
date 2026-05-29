from pydantic import BaseModel
from typing import List, Optional


class TranscricaoResponse(BaseModel):
    id_marcador: Optional[int] = None
    id_deputado: Optional[int] = None
    nome_deputado: Optional[str] = None
    texto: Optional[str] = None
    id_video: Optional[str] = None
    tempo_inicial: Optional[float] = None
    tempo_final: Optional[float] = None
    data_inclusao: Optional[str] = None
    ids_segmentos: Optional[str] = None
    chave_fase: Optional[str] = None
    titulo_fase: Optional[str] = None
    sentimento: Optional[str] = None
    tom_discurso: Optional[str] = None
    temas: Optional[str] = None
    num_palavras: Optional[int] = None
    cosine_distance: float


class SearchResponse(BaseModel):
    results: List[TranscricaoResponse]
