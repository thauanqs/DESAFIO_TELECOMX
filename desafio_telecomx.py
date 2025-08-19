import pandas as pd
url = 'https://raw.githubusercontent.com/ingridcristh/challenge2-data-science/refs/heads/main/TelecomX_Data.json'
dados = pd.read_json(url)

print(dados.head()) #mostra as primeiras linhas - tail() mostra as ultimas