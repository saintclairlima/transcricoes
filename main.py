from fastapi import FastAPI

from transcricoes.controller import router

app = FastAPI(
    title="Doxxo Transcrições PGVector API"
)

app.include_router(router)
