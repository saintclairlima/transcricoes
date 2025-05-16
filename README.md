# Fragmentação Semântica de Transcrições de Vídeos da ALERN

## Crédito
O projeto é uma adaptação e implementação do projeto descrito no artigo https://medium.com/data-science/text-tiling-done-right-building-solid-foundations-for-your-personal-llm-e70947779ac1

O código original pode ser visto no repositório do autor (https://github.com/massi82/texttiling/)

## Utilização
``` bash
git clone https://github.com/saintclairlima/transcricoes.git
cd /content/transcricoes
pip install -r requirements.txt
python -m spacy download pt_core_news_sm
python -m src.processamento --url_transcricoes "./dados/videos_transcricao.json"
```