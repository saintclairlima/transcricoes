import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")

DATABASE_URL = (
    f"postgresql://postgres.vzhighpgtjidzqagnesl:"
    f"{SUPABASE_PASSWORD}"
    "@aws-1-us-east-1.pooler.supabase.com:6543/postgres"
)

EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K_RESULTS = 10
