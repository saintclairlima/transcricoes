import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = (
    f"postgresql://postgres.vzhighpgtjidzqagnesl:"
    f"{SUPABASE_PASSWORD}"
    "@aws-1-us-east-1.pooler.supabase.com:6543/postgres"
)
EMBEDDING_MODEL = "gemini-embedding-2"
TOP_K_RESULTS = 10
