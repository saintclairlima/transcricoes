from pydantic import BaseModel
from typing import List, Optional


class TranscricaoResponse(BaseModel):
    id_marcador: Optional[int] = None
    id_deputado: Optional[int] = None
    nome_deputado: Optional[str] = None
    texto: Optional[str] = None
    
    # Corrigido: O banco retorna um número inteiro para o ID do vídeo
    id_video: Optional[int] = None
    
    # Corrigido: Como vem '01:25:49.334', o tipo correto aqui para o Pydantic aceitar é str
    tempo_inicial: Optional[str] = None
    tempo_final: Optional[str] = None
    
    data_inclusao: Optional[str] = None
    
    # Corrigido: O banco retorna um array/lista de números inteiros
    ids_segmentos: Optional[List[int]] = None
    
    chave_fase: Optional[str] = None
    titulo_fase: Optional[str] = None
    sentimento: Optional[str] = None
    tom_discurso: Optional[str] = None
    
    # Corrigido: O banco retorna uma lista de dicionários estruturados
    temas: Optional[List[TemaSchema]] = None
    
    num_palavras: Optional[int] = None
    cosine_distance: float


class SearchResponse(BaseModel):
    results: List[TranscricaoResponse]
