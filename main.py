from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from controller import router

app = FastAPI(
    title="API de Busca Semântica mas Transcrições de Sessões Legislativas"
)

# origins = [
#     "http://localhost:4200",
#     "http://127.0.0.1:4200"
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],            # Allows listed origins
    allow_credentials=False,           # Allows cookies/authentication headers
    allow_methods=["*"],              # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],              # Allows all custom headers
)


app.include_router(router)