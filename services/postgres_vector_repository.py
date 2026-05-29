from psycopg2.extras import RealDictCursor
from database.connection import PostgresConnection

class PostgresVectorRepository:

    def search_similar_documents(
        self,
        embedding,
        top_k: int = 10
    ):

        conn = None
        cur = None

        try:
            conn = PostgresConnection.get_connection()
            
            cur = conn.cursor(cursor_factory=RealDictCursor)

            query = """
            SELECT
                idMarcador,
                idDeputado,
                nomeDeputado,
                texto,
                idVideo,
                tempoInicial,
                tempoFinal,
                dataInclusao,
                idsSegmentos,
                chaveFase,
                tituloFase,
                sentimento,
                tom_do_discurso,
                temas,
                word_count,
                embeddings <=> %s AS cosine_distance
            FROM transcricao
            ORDER BY cosine_distance ASC
            LIMIT %s;
            """

            cur.execute(query, (embedding, top_k))

            results = cur.fetchall()

            return results

        finally:
            if cur:
                cur.close()

            if conn:
                conn.close()
