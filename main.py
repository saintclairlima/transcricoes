from fastapi import FastAPI

from controller import router

app = FastAPI(
    title="API de Busca Semântica mas Transcrições de Sessões Legislativas"
)

app.include_router(router)
