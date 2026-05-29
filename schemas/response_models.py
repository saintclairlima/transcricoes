from pydantic import BaseModel
from typing import List, Optional


class TranscricaoResponse(BaseModel):
    idMarcador: Optional[int]
    idDeputado: Optional[int]
    nomeDeputado: Optional[str]
    texto: Optional[str]
    idVideo: Optional[str]
    tempoInicial: Optional[float]
    tempoFinal: Optional[float]
    dataInclusao: Optional[str]
    idsSegmentos: Optional[str]
    chaveFase: Optional[str]
    tituloFase: Optional[str]
    sentimento: Optional[str]
    tomDiscurso: Optional[str]
    temas: Optional[str]
    numPalavras: Optional[int]
    cosine_distance: float


class SearchResponse(BaseModel):
    results: List[TranscricaoResponse]
