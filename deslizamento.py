import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

dados = pd.read_csv('./casosEstudo.csv')

dados

dados.head()

x = dados[["umidade_(%)", "qtd_chuva_(3_dias)", "inclinacao_em_graus", "profundidade_do_solo", "solo_argiloso", "cobertura_vegetal"]]
y = dados["deslizamento"]

SEED = 487

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state = SEED, stratify=y)

print(f'Treinaremos com {len(treino_x)} elementos')
print(f'Testaremos com {len(teste_x)} elementos')

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)

previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print(f"A acurácia foi de {acuracia:.2f}%")