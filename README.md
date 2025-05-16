# Fragmentação Semântica de Transcrições de Vídeos da ALERN

## Crédito
O projeto é uma adaptação e implementação do projeto descrito no artigo https://medium.com/data-science/text-tiling-done-right-building-solid-foundations-for-your-personal-llm-e70947779ac1

O código original pode ser visto no repositório do autor (https://github.com/massi82/texttiling/)

## Descrição do conteúdo
### Código
Em `src` temos o arquivo `fragmentador-semantico.py`, que contém a lógica de particionamento de texto com base em tópicos, e `gerar-fragmentos-transcricoes.py` que recebe um conjunto de dados salvos em formato `.json` (como o que está em `dados/videos_transcricao.json` e aplica o particionamento semântico.

Em `src/testes` tem código e dados utilizados para avaliação de alguns modelos utilizados para efetuação do particionamento. 

### Dados

Em `dados` temos um arquivo `.json` com os dados públicos das transcrições conforme disponibilizadas na página de vídeos da Assembleia Legislativa do Estado do Rio Grande do Norte. São os dados que são utilizados para geração de fragmentos no código de exemplo. 

Além disso

## Utilização
``` bash
git clone https://github.com/saintclairlima/transcricoes.git
cd /content/transcricoes
pip install -r requirements.txt
python -m spacy download pt_core_news_sm
python -m src.gerar-fragmentos-transcricoes --url_transcricoes "./dados/videos_transcricao.json"
```
