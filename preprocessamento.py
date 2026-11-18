"""
Pré-processamento compartilhado entre o treinamento e a inferência.

Treino e inferência PRECISAM aplicar exatamente as mesmas transformações.
Por isso elas ficam neste módulo, importado pelos dois scripts.

Base: Breast Cancer (UCI) - https://archive.ics.uci.edu/dataset/14/breast+cancer
"""
import numpy as np
import pandas as pd

# Nomes das colunas conforme a descrição da UCI
ATRIBUTOS = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps',
             'deg-malig', 'breast', 'breast-quad', 'irradiat']
CLASSE = 'Class'

# Domínio de cada atributo segundo a descrição da UCI
# (inclui faixas que não aparecem nos 286 registros, mas que são válidas)
VALORES_VALIDOS = {
    'age': ['10-19', '20-29', '30-39', '40-49', '50-59',
            '60-69', '70-79', '80-89', '90-99'],
    'menopause': ['lt40', 'ge40', 'premeno'],
    'tumor-size': ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29',
                   '30-34', '35-39', '40-44', '45-49', '50-54', '55-59'],
    'inv-nodes': ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20',
                  '21-23', '24-26', '27-29', '30-32', '33-35', '36-39'],
    'node-caps': ['yes', 'no'],
    'deg-malig': ['1', '2', '3'],
    'breast': ['left', 'right'],
    'breast-quad': ['left_up', 'left_low', 'right_up', 'right_low', 'central'],
    'irradiat': ['yes', 'no'],
}

# Atributos que são faixas numéricas ordenadas: viram o ponto médio da faixa
ATRIBUTOS_FAIXA = ['age', 'tumor-size', 'inv-nodes']
# Atributos binários: yes/no e left/right viram 0/1
ATRIBUTOS_BINARIOS = {'node-caps': 'yes', 'irradiat': 'yes', 'breast': 'right'}
# Atributos nominais (sem ordem): viram colunas 0/1 (one-hot)
ATRIBUTOS_NOMINAIS = ['menopause', 'breast-quad']


def carregar_dados(caminho):
    """Lê o CSV como texto e converte o marcador de ausente '?' em NaN."""
    dados = pd.read_csv(caminho, sep=',', dtype=str)
    dados = dados.apply(lambda coluna: coluna.str.strip())
    return dados.replace('?', np.nan)


def calcular_modas(atributos):
    """Moda (valor mais frequente) de cada atributo, usada para imputar ausentes."""
    return {coluna: atributos[coluna].mode()[0] for coluna in ATRIBUTOS}


def imputar_ausentes(atributos, modas):
    """Substitui valores ausentes pela moda calculada no conjunto de TREINO."""
    return atributos.fillna(value=modas)


def ponto_medio(faixa):
    """'40-49' -> 44.5"""
    inicio, fim = faixa.split('-')
    return (int(inicio) + int(fim)) / 2


def codificar(atributos):
    """
    Converte os atributos categóricos em numéricos, sempre com as mesmas
    colunas e na mesma ordem (requisito para o normalizador e o modelo).
    """
    codificados = pd.DataFrame(index=atributos.index)
    for coluna in ATRIBUTOS_FAIXA:
        codificados[coluna] = atributos[coluna].map(ponto_medio)
    codificados['deg-malig'] = atributos['deg-malig'].astype(int)
    for coluna, valor_positivo in ATRIBUTOS_BINARIOS.items():
        codificados[coluna] = (atributos[coluna] == valor_positivo).astype(int)
    for coluna in ATRIBUTOS_NOMINAIS:
        for categoria in VALORES_VALIDOS[coluna]:
            codificados[f'{coluna}_{categoria}'] = (atributos[coluna] == categoria).astype(int)
    return codificados


def montar_paciente(paciente):
    """
    Valida os dados de uma nova paciente e devolve um DataFrame de 1 linha.
    Aceita '?' (ou ausência do campo) como valor desconhecido; ele será
    imputado com a moda do treino.
    """
    erros = []
    linha = {}
    for coluna in ATRIBUTOS:
        valor = paciente.get(coluna)
        valor = None if valor is None else str(valor).strip().lower()
        if coluna == 'breast-quad' and valor:
            valor = valor.replace('-', '_')   # a UCI escreve left-up; o CSV usa left_up
        if valor in (None, '', '?'):
            linha[coluna] = np.nan
        elif valor in VALORES_VALIDOS[coluna]:
            linha[coluna] = valor
        else:
            erros.append(f"  {coluna}: '{valor}' inválido. Valores aceitos: "
                         f"{', '.join(VALORES_VALIDOS[coluna])}")
    if erros:
        raise ValueError('Dados da paciente inválidos:\n' + '\n'.join(erros))
    return pd.DataFrame([linha], columns=ATRIBUTOS)
