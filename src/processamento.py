import argparse
import json
from typing import List

from src.fragmentador_semantico import fragmentar as fragmentacao_semantica

def fragmentar_texto(texto: str, nome_modelo:str) -> List[str]:
    return fragmentacao_semantica(texto=texto, modelo=nome_modelo)

def fragmentar_transcricoes(url_transcricoes: str, url_saida, nome_modelo, log=True):
    if log: print(f'Iniciando fragmentação de {url_transcricoes}. Saída em: {url_saida}')

    with open(url_transcricoes, 'r', encoding='utf-8') as arq:
        transcricoes = json.load(arq)

    num_entradas = len(transcricoes)
    for idx in range(num_entradas):
        if log: print(f'\rProcessando transcrições - {format((idx+1)/num_entradas,'.2%')}', end='')
        transcricao = transcricoes[idx]
        transcricao['transcricao'] = fragmentar_texto(texto=transcricao['transcricao'], nome_modelo=nome_modelo)
    if log: print('\nConcluído.')
    if log: print(f'Salvando resultados em {url_saida}')
    with open(url_saida, 'w', encoding='utf-8') as arq:
        json.dump(transcricoes, arq, ensure_ascii=False, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Gera um arquivo de transcrições com fragmentaçaõ semântica")
    parser.add_argument('--url_transcricoes', type=str, required=True, help="caminho para json estruturado com as transcrições")
    parser.add_argument('--url_saida', type=str, help="caminho onde salvar o json resultante")
    parser.add_argument('--nome_modelo', type=str, help="modelo a ser usado para fragmentação")

    args = parser.parse_args()
    url_transcricoes = args.url_transcricoes
    url_saida = f'{args.url_transcricoes[:-5]}_fragmentado.json' if not args.url_saida else args.url_saida
    nome_modelo = 'jvanhoof/all-MiniLM-L6-multilingual-v2-en-es-pt-pt-br-v2' if not args.nome_modelo else args.nome_modelo

    fragmentar_transcricoes(url_transcricoes, url_saida, nome_modelo)