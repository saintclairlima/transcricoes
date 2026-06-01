import ast
from typing import List, Optional
import pandas as pd


class DeputadoTemaService:

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = self._carregar_e_tratar_dados()

    def _carregar_e_tratar_dados(self) -> pd.DataFrame:
        """Carrega o CSV e converte a string de temas para listas reais do Python."""
        try:
            df = pd.read_csv(self.csv_path)

            def converter_temas(valor):
                if pd.isna(valor) or str(valor).strip() == "":
                    return []
                try:
                    return ast.literal_eval(valor)
                except Exception:
                    return []

            df["temas"] = df["temas"].apply(converter_temas)
            print(
                f"✅ Service: {len(df)} instâncias carregadas com sucesso na memória!"
            )
            return df
        except Exception as e:
            print(f"❌ Service: Erro ao carregar arquivo CSV: {e}")
            return pd.DataFrame()

    def buscar_discursos(
        self,
        id_deputado: Optional[int] = None,
        nome_deputado: Optional[str] = None,
        tema: Optional[str] = None,
    ) -> List[dict]:
        """Aplica os filtros solicitados no DataFrame e retorna uma lista de dicionários."""
        if self.df.empty:
            return []

        resultado = self.df.copy()

        # Filtro por ID do Deputado
        if id_deputado is not None:
            resultado = resultado[resultado["idDeputado"] == id_deputado]

        # Filtro por Nome do Deputado (Busca parcial/case insensitive)
        if nome_deputado:
            resultado = resultado[
                resultado["nomeDeputado"].str.contains(
                    nome_deputado, case=False, na=False
                )
            ]

        # Filtro por Tema dentro da estrutura de lista/dicionário
        if tema:
            tema_procurado = tema.strip().lower()

            def contem_tema(lista_de_temas):
                for t in lista_de_temas:
                    if t.get("nome", "").lower() == tema_procurado:
                        return True
                return False

            mascara_tema = resultado["temas"].apply(contem_tema)
            resultado = resultado[mascara_tema]

        # Limpa valores nulos para evitar quebras no JSON e exporta
        resultado = resultado.fillna(value="")
        return resultado.to_dict(orient="records")