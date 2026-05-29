import os
from dotenv import load_dotenv

load_dotenv()

SENHA_SUPABASE = os.getenv("SENHA_SUPABASE")
CHAVE_API_GEMINI = os.getenv("GEMINI_API_KEY")
DATABASE_URL = (
    f"postgresql://postgres.vzhighpgtjidzqagnesl:"
    f"{SENHA_SUPABASE}"
    "@aws-1-us-east-1.pooler.supabase.com:6543/postgres"
)
MODELO_EMBEDDINGS = "gemini-embedding-2"
TOP_K_RESULTS = 10
