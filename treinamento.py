"""
MÓDULO DE TREINAMENTO - Breast Cancer (UCI)
Objetivo: prever o atributo Class (recurrence-events / no-recurrence-events)

Procedimentos (mesma sequência vista em sala):
 1. Carregar os dados
 2. Separar atributos e classe
 3. Separar treino e teste (o teste só é usado na avaliação final)
 4. Tratar valores ausentes ('?')
 5. Codificar os atributos categóricos em numéricos
 6. Hiperparametrizar e comparar vários algoritmos classificadores
    (balanceamento SMOTE + normalização MinMax + RandomizedSearchCV),
    avaliando cada algoritmo com e sem o balanceamento
 7. Escolher o modelo de MELHOR ACURÁCIA (revalidação cruzada no treino)
    e avaliá-lo no conjunto de teste
 8. Treinamento definitivo com todos os dados e gravação dos modelos
"""
import json
import os
from pickle import dump

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE               # balancear os dados
from imblearn.pipeline import Pipeline                 # SMOTE dentro da validação cruzada
from sklearn.preprocessing import MinMaxScaler          # normalizar os dados
from sklearn.model_selection import (train_test_split, RandomizedSearchCV,
                                     RepeatedStratifiedKFold, cross_val_score)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.base import clone

import preprocessamento as pp

SEMENTE = 42

# O balanceamento (SMOTE) é avaliado como uma opção: cada algoritmo é testado
# COM e SEM balanceamento, e vence a combinação de maior acurácia.
# Para obrigar o balanceamento em todos os modelos, use: OPCOES_BALANCEAMENTO = [True]
OPCOES_BALANCEAMENTO = [True, False]

PASTA = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_DADOS = os.path.join(PASTA, 'dados', 'breast-cancer.csv')
PASTA_MODELOS = os.path.join(PASTA, 'modelos')
PASTA_RESULTADOS = os.path.join(PASTA, 'resultados')
os.makedirs(PASTA_MODELOS, exist_ok=True)
os.makedirs(PASTA_RESULTADOS, exist_ok=True)

# 1. CARREGAR OS DADOS ------------------------------------------------------
dados = pp.carregar_dados(ARQUIVO_DADOS)
print(f'Registros: {dados.shape[0]}  |  Atributos: {dados.shape[1] - 1}')
print('Valores ausentes por atributo:')
print(dados.isna().sum()[dados.isna().sum() > 0].to_string())
print('\nDistribuição da classe:')
print(dados[pp.CLASSE].value_counts().to_string())

# 2. SEPARAR ATRIBUTOS E CLASSE ---------------------------------------------
dados_atributos = dados[pp.ATRIBUTOS]
dados_classes = dados[pp.CLASSE]

# 3. SEPARAR TREINO E TESTE -------------------------------------------------
# 70% treino / 30% teste, estratificado para manter a proporção das classes.
# O teste fica guardado e não participa de nenhuma decisão do treinamento.
atributos_treino, atributos_teste, classes_treino, classes_teste = train_test_split(
    dados_atributos, dados_classes, test_size=0.30,
    stratify=dados_classes, random_state=SEMENTE)
print(f'\nTreino: {len(atributos_treino)} registros  |  Teste: {len(atributos_teste)} registros')

# 4. TRATAR VALORES AUSENTES ------------------------------------------------
# A moda é calculada só no treino e aplicada ao treino e ao teste.
modas_treino = pp.calcular_modas(atributos_treino)
atributos_treino = pp.imputar_ausentes(atributos_treino, modas_treino)
atributos_teste = pp.imputar_ausentes(atributos_teste, modas_treino)

# 5. CODIFICAR OS ATRIBUTOS CATEGÓRICOS -------------------------------------
atributos_treino = pp.codificar(atributos_treino)
atributos_teste = pp.codificar(atributos_teste)
print(f'Atributos após a codificação: {atributos_treino.shape[1]}')

# 6. HIPERPARAMETRIZAÇÃO E COMPARAÇÃO DE ALGORITMOS -------------------------
# Para cada algoritmo, o RandomizedSearchCV testa combinações de hiperparâmetros.
# O balanceamento (SMOTE) e a normalização (MinMax) ficam DENTRO do pipeline:
# em cada rodada da validação cruzada eles são ajustados só com a parte de
# treino da rodada, sem dados sintéticos vazando para a parte de validação.
# Cada algoritmo é avaliado com e sem balanceamento (ver OPCOES_BALANCEAMENTO).
algoritmos = {
    'DecisionTree': (
        DecisionTreeClassifier(random_state=SEMENTE),
        {'criterion': ['gini', 'entropy', 'log_loss'],
         'splitter': ['best', 'random'],
         'max_depth': [None, 2, 3, 4, 5, 6, 8, 10],
         'min_samples_split': [int(x) for x in np.linspace(start=2, stop=20, num=10)],
         'min_samples_leaf': [1, 2, 4, 6, 8, 10]}),
    'RandomForest': (
        RandomForestClassifier(random_state=SEMENTE),
        {'n_estimators': [int(x) for x in np.linspace(start=50, stop=500, num=10)],
         'criterion': ['gini', 'entropy', 'log_loss'],
         'max_depth': [None, 3, 4, 5, 6, 8, 10],
         'min_samples_split': [2, 4, 6, 8, 10],
         'min_samples_leaf': [1, 2, 4, 6, 8],
         'max_features': ['sqrt', 'log2', None]}),
    'GradientBoosting': (
        GradientBoostingClassifier(random_state=SEMENTE),
        {'n_estimators': [50, 100, 150, 200, 300],
         'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
         'max_depth': [1, 2, 3, 4],
         'subsample': [0.6, 0.8, 1.0],
         'min_samples_leaf': [1, 3, 5, 10]}),
    'KNN': (
        KNeighborsClassifier(),
        {'n_neighbors': list(range(1, 32, 2)),
         'weights': ['uniform', 'distance'],
         'p': [1, 2]}),
    'SVM': (
        SVC(random_state=SEMENTE),
        {'C': [float(c) for c in np.logspace(-2, 2, 9)],
         'kernel': ['linear', 'rbf', 'poly'],
         'gamma': ['scale', 'auto', 0.01, 0.1, 1.0]}),
    'LogisticRegression': (
        LogisticRegression(solver='liblinear', max_iter=1000, random_state=SEMENTE),
        {'C': [float(c) for c in np.logspace(-3, 2, 11)],
         'l1_ratio': [0.0, 1.0]}),   # 0.0 = penalidade L2 | 1.0 = penalidade L1
    'NaiveBayes': (
        GaussianNB(),
        {'var_smoothing': [float(v) for v in np.logspace(-9, 0, 10)]}),
}

# Validação cruzada estratificada com 5 partes, repetida 3 vezes
validacao = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEMENTE)

# Referência mínima: um "modelo" que sempre responde a classe majoritária.
# Um classificador só é útil se superar essa acurácia.
referencia = DummyClassifier(strategy='most_frequent')
acuracia_referencia_cv = cross_val_score(referencia, atributos_treino, classes_treino,
                                         cv=validacao, scoring='accuracy').mean()
referencia.fit(atributos_treino, classes_treino)
acuracia_referencia = accuracy_score(classes_teste, referencia.predict(atributos_teste))
print(f'\nAcurácia de referência (sempre a classe majoritária): '
      f'CV = {acuracia_referencia_cv:.4f}  |  teste = {acuracia_referencia:.4f}\n')

resultados = []
buscas = {}
for nome, (classificador, grade) in algoritmos.items():
    for balancear in OPCOES_BALANCEAMENTO:
        etapas = [('balanceador', SMOTE(random_state=SEMENTE))] if balancear else []
        etapas += [('normalizador', MinMaxScaler()),
                   ('classificador', classificador)]
        pipeline = Pipeline(etapas)
        grade_pipeline = {f'classificador__{parametro}': valores
                          for parametro, valores in grade.items()}
        n_combinacoes = int(np.prod([len(v) for v in grade.values()]))
        busca = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=grade_pipeline,
            n_iter=min(40, n_combinacoes),
            cv=validacao,
            scoring='accuracy',
            random_state=SEMENTE,
            n_jobs=-1,
        )
        busca.fit(atributos_treino, classes_treino)
        buscas[(nome, balancear)] = busca

        acuracia_cv = busca.best_score_
        desvio_cv = busca.cv_results_['std_test_score'][busca.best_index_]
        acuracia_teste = accuracy_score(classes_teste, busca.predict(atributos_teste))
        resultados.append({'algoritmo': nome,
                           'balanceamento_smote': 'sim' if balancear else 'não',
                           'acuracia_validacao_cruzada': round(float(acuracia_cv), 4),
                           'desvio_validacao_cruzada': round(float(desvio_cv), 4),
                           'acuracia_teste': round(float(acuracia_teste), 4)})
        print(f'{nome:<19} SMOTE={"sim" if balancear else "não"}  '
              f'acurácia CV = {acuracia_cv:.4f} (±{desvio_cv:.4f})'
              f'  |  acurácia teste = {acuracia_teste:.4f}')

# 7. ESCOLHER O MODELO DE MELHOR ACURÁCIA -----------------------------------
# A acurácia da busca é otimista: entre 40 combinações, a escolhida tende a ser
# a que teve "sorte" naquelas divisões. Por isso, a melhor configuração de cada
# algoritmo é REVALIDADA em uma validação cruzada nova (outras divisões, 10
# repetições), ainda só com o treino. Vence a maior acurácia na revalidação.
# O conjunto de teste continua fora da escolha.
revalidacao = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEMENTE + 1)
for resultado in resultados:
    busca = buscas[(resultado['algoritmo'], resultado['balanceamento_smote'] == 'sim')]
    notas = cross_val_score(clone(busca.best_estimator_), atributos_treino, classes_treino,
                            cv=revalidacao, scoring='accuracy', n_jobs=-1)
    resultado['acuracia_revalidacao'] = round(float(notas.mean()), 4)
    resultado['desvio_revalidacao'] = round(float(notas.std()), 4)

comparacao = pd.DataFrame(resultados).sort_values(
    ['acuracia_revalidacao', 'acuracia_validacao_cruzada'], ascending=False).reset_index(drop=True)
comparacao.to_csv(os.path.join(PASTA_RESULTADOS, 'comparacao_modelos.csv'), index=False)
print('\nRanking dos algoritmos:')
print(comparacao.to_string(index=False))

melhor_nome = comparacao.loc[0, 'algoritmo']
melhor_balancear = comparacao.loc[0, 'balanceamento_smote'] == 'sim'
melhor_busca = buscas[(melhor_nome, melhor_balancear)]
melhores_parametros = {parametro.replace('classificador__', ''): valor
                       for parametro, valor in melhor_busca.best_params_.items()}
classes_preditas = melhor_busca.predict(atributos_teste)
acuracia_teste = accuracy_score(classes_teste, classes_preditas)

print(f'\nMELHOR MODELO: {melhor_nome} | balanceamento SMOTE: '
      f'{"sim" if melhor_balancear else "não"}')
print(f'Hiperparâmetros: {melhores_parametros}')
print(f'Acurácia na revalidação cruzada: {comparacao.loc[0, "acuracia_revalidacao"]:.4f} '
      f'(referência: {acuracia_referencia_cv:.4f})')
print(f'Acurácia no teste: {acuracia_teste:.4f} (referência: {acuracia_referencia:.4f})')
rotulos = ['no-recurrence-events', 'recurrence-events']
print('Matriz de confusão (linhas = real, colunas = previsto):')
print(pd.DataFrame(confusion_matrix(classes_teste, classes_preditas, labels=rotulos),
                   index=rotulos, columns=rotulos).to_string())
print(classification_report(classes_teste, classes_preditas, zero_division=0))

# 8. TREINAMENTO DEFINITIVO COM TODOS OS DADOS ------------------------------
# Com o algoritmo e os hiperparâmetros escolhidos, treina-se de novo usando
# as 286 pacientes, seguindo a sequência da aula: balancear (se a opção
# vencedora usa SMOTE), normalizar, salvar o normalizador, treinar e salvar.
modas = pp.calcular_modas(dados_atributos)
dados_atributos_codificados = pp.codificar(pp.imputar_ausentes(dados_atributos, modas))

# 8.1 Balancear
if melhor_balancear:
    resampler = SMOTE(random_state=SEMENTE)
    dados_atributos_b, dados_classes_b = resampler.fit_resample(dados_atributos_codificados,
                                                                dados_classes)
    print(f'Após o balanceamento: {dados_classes_b.value_counts().to_dict()}')
else:
    dados_atributos_b, dados_classes_b = dados_atributos_codificados, dados_classes
    print('Modelo vencedor sem balanceamento: treino definitivo com as classes originais.')

# 8.2 Normalizar (o normalizador é treinado e salvo para a inferência)
normalizador = MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_atributos_b)
dump(modelo_normalizador,
     open(os.path.join(PASTA_MODELOS, 'normalizador_breast_cancer.model'), 'wb'))
dados_atributos_normalizados = pd.DataFrame(modelo_normalizador.transform(dados_atributos_b),
                                            columns=dados_atributos_codificados.columns)

# 8.3 Treinar o classificador com a arquitetura otimizada e salvar
classificador = algoritmos[melhor_nome][0].set_params(**melhores_parametros)
modelo_breast_cancer = classificador.fit(dados_atributos_normalizados, dados_classes_b)
dump(modelo_breast_cancer,
     open(os.path.join(PASTA_MODELOS, 'breast_cancer_classificador.model'), 'wb'))

# 8.4 Salvar o que a inferência precisa (modas para ausentes) e o resumo
metadados = {
    'algoritmo': melhor_nome,
    'balanceamento_smote': bool(melhor_balancear),
    'hiperparametros': {k: (v.item() if hasattr(v, 'item') else v)
                        for k, v in melhores_parametros.items()},
    'acuracia_validacao_cruzada_busca': float(comparacao.loc[0, 'acuracia_validacao_cruzada']),
    'acuracia_revalidacao_cruzada': float(comparacao.loc[0, 'acuracia_revalidacao']),
    'acuracia_teste': round(float(acuracia_teste), 4),
    'acuracia_referencia_classe_majoritaria_cv': round(float(acuracia_referencia_cv), 4),
    'acuracia_referencia_classe_majoritaria_teste': round(float(acuracia_referencia), 4),
    'modas_para_valores_ausentes': modas,
    'colunas_codificadas': list(dados_atributos_codificados.columns),
}
with open(os.path.join(PASTA_MODELOS, 'metadados.json'), 'w', encoding='utf-8') as arquivo:
    json.dump(metadados, arquivo, indent=2, ensure_ascii=False)

print('\nModelos salvos em modelos/:')
print('  normalizador_breast_cancer.model')
print('  breast_cancer_classificador.model')
print('  metadados.json')
