# Transcrições - API de Busca Semântica com PGVector

API para pesquisa semântica em transcrições utilizando:

- FastAPI
- Google Gemini Embeddings
- PostgreSQL
- PGVector

A aplicação recebe um texto, gera embeddings vetoriais utilizando Gemini e realiza busca por similaridade em registros armazenados no PostgreSQL.

---

## Arquitetura

Cliente
↓
FastAPI
↓
Gemini Embeddings
↓
PGVector
↓
Tabela transcricao

---

## Requisitos

Python 3.10+

---

## Instalação

Clone o projeto:

git clone https://github.com/saintclairlima/transcricoes.git

Instale dependências:

pip install -r requirements.txt

---

## Configuração

Crie um arquivo `.env`:

SENHA_SUPABASE=
GEMINI_API_KEY=

---

## Executar

uvicorn main:app --host 0.0.0.0 --port 8000

---

## Swagger

http://localhost:8000/docs

---

## Endpoint POST

POST /search

Body:

{
  "query": "reforma tributária",
  "top_k": 10
}

---

## Endpoint GET

GET /search?query=reforma tributária&top_k=10

---

## Resposta

{
  "results": [...]
}

---

## Campos retornados

- id_marcador
- id_deputado
- nome_deputado
- texto
- id_video
- tempo_inicial
- tempo_final
- data_inclusao
- ids_segmentos
- chave_fase
- titulo_fase
- sentimento
- tom_discurso
- temas
- num_palavras
- cosine_distance

---

## Estrutura do Projeto

config/
database/
schemas/
services/

main.py
controller.py

---

## Busca Vetorial

A busca utiliza o operador PGVector:

embeddings <=> query_embedding

Os resultados são ordenados por distância de cosseno crescente.

Quanto menor a distância, maior a similaridade semântica.

---

## Tecnologias

- FastAPI
- PostgreSQL
- PGVector
- Gemini Embeddings
- Pydantic
- Psycopg2
