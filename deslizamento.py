import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def deslizamento(caso_estudo):
  if (modelo.predict([caso_estudo])) == 1:
    return ("Cuidado! Pode haver um deslizamento!")
  else:
    return ("Não há risco de deslizamento!")

# coletando os dados
dados = pd.read_csv('./casosEstudo.csv')

print(dados)
print(dados.head())

# separando as variáveis dos resultados esperados
x = dados[["umidade_(%)", "qtd_chuva_(3_dias)", "inclinacao_em_graus", "profundidade_do_solo", "solo_argiloso", "cobertura_vegetal"]]
y = dados["deslizamento"]

# treinando o modelo
SEED = 487

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state = SEED, stratify=y)

print(f'Treinaremos com {len(treino_x)} elementos')
print(f'Testaremos com {len(teste_x)} elementos')

modelo = LinearSVC()
modelo.fit(treino_x.values, treino_y.values)

previsoes = modelo.predict(teste_x.values)

acuracia = accuracy_score(teste_y.values, previsoes) * 100
print(f"A acurácia foi de {acuracia:.2f}%")

# testes caso de estudo -> São Sebastião 2023

# umidade (%): 100 -> solo saturado (https://g1.globo.com/sp/vale-do-paraiba-regiao/noticia/2023/02/20/liquefacao-do-solo-provocou-deslizamentos-no-litoral-de-sp-entenda-fenomeno.ghtml)
# qtd_chuva (3 dias): 683 (15 horas) https://g1.globo.com/sp/vale-do-paraiba-regiao/noticia/2023/02/20/liquefacao-do-solo-provocou-deslizamentos-no-litoral-de-sp-entenda-fenomeno.ghtml
# inclinacao_em_graus: 30 https://g1.globo.com/sp/sao-paulo/noticia/2023/02/22/deslizamentos-frequentes-na-serra-do-mar-sao-resultado-de-solo-superficial-alta-inclinacao-e-paredao-de-nuvens.ghtml
# profundidade_solo: 2 https://g1.globo.com/sp/sao-paulo/noticia/2023/02/22/deslizamentos-frequentes-na-serra-do-mar-sao-resultado-de-solo-superficial-alta-inclinacao-e-paredao-de-nuvens.ghtml
# solo_argiloso: 0 (https://fflorestal.sp.gov.br/wp-content/uploads/2022/07/plano_de_manejo_arie_sao_sebastiao-3.pdf)
# cobertura_vegetal: 1 (fotos do local)

caso_estudo = [100, 683, 30, 2, 0, 1]
print("\n\n----- Caso São Sebastião 2023 -----")
print(deslizamento(caso_estudo))

# Testes caso de estudo - Petrópolis 2022

# umidade (%): 100 - saturado - https://www.periodicos.rc.biblioteca.unesp.br/index.php/geociencias/article/view/17210/12759
# qtd_chuva (3 dias): 259mm (6 horas) - https://www.periodicos.rc.biblioteca.unesp.br/index.php/geociencias/article/view/17210/12759
# inclinacao_em_graus: 33 (morro da oficina - maior parte de 20 a 45) - https://schenautomacao.com.br/cbge2022/envio/files/trabalho1_187.pdf
# profundidade_solo: 4.5m - https://schenautomacao.com.br/cbge2022/envio/files/trabalho1_187.pdf
# solo_argiloso: 1 - https://schenautomacao.com.br/cbge2022/envio/files/trabalho1_187.pdf
# cobertura_vegetal: 1 - https://www.periodicos.rc.biblioteca.unesp.br/index.php/geociencias/article/view/17210/12759

caso_estudo = [100, 259, 33, 4.5, 1, 1]
print("\n\n----- Caso Petrópolis 2022 -----")
print(deslizamento(caso_estudo))
