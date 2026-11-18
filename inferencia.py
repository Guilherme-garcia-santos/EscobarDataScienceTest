"""
MÓDULO DE INFERÊNCIA - Breast Cancer (UCI)
Recebe uma nova paciente e determina o indicativo de diagnóstico (atributo Class):
  recurrence-events     -> indicativo de recorrência do câncer
  no-recurrence-events  -> indicativo de não recorrência

Uso:
  python inferencia.py                       (classifica a paciente de exemplo)
  python inferencia.py --age 40-49 --menopause premeno --tumor-size 30-34 \
      --inv-nodes 3-5 --node-caps yes --deg-malig 3 --breast left \
      --breast-quad left_low --irradiat no

Um atributo desconhecido pode ser informado como '?'. Ele é preenchido com a
moda (valor mais frequente) observada no treinamento.

Antes da inferência, execute o treinamento (python treinamento.py) ao menos uma vez.
"""
import argparse
import json
import os
from pickle import load

import pandas as pd

import preprocessamento as pp

PASTA = os.path.dirname(os.path.abspath(__file__))
PASTA_MODELOS = os.path.join(PASTA, 'modelos')

# Paciente de exemplo, usada quando nenhum dado é passado pela linha de comando
NOVA_PACIENTE_EXEMPLO = {
    'age': '40-49',
    'menopause': 'premeno',
    'tumor-size': '30-34',
    'inv-nodes': '3-5',
    'node-caps': 'yes',
    'deg-malig': '3',
    'breast': 'left',
    'breast-quad': 'left_low',
    'irradiat': 'no',
}

DESCRICAO_CLASSE = {
    'recurrence-events': 'indicativo de RECORRÊNCIA do câncer de mama',
    'no-recurrence-events': 'indicativo de NÃO RECORRÊNCIA do câncer de mama',
}


def carregar_modelos():
    """Carrega o normalizador, o classificador e os metadados salvos no treinamento."""
    normalizador = load(open(os.path.join(PASTA_MODELOS, 'normalizador_breast_cancer.model'), 'rb'))
    modelo = load(open(os.path.join(PASTA_MODELOS, 'breast_cancer_classificador.model'), 'rb'))
    with open(os.path.join(PASTA_MODELOS, 'metadados.json'), encoding='utf-8') as arquivo:
        metadados = json.load(arquivo)
    return normalizador, modelo, metadados


def diagnosticar(paciente):
    """
    Recebe um dicionário com os 9 atributos da paciente e devolve
    (classe prevista, probabilidade de cada classe).
    """
    normalizador, modelo, metadados = carregar_modelos()

    # 1. Validar e montar a paciente no mesmo formato da base
    nova_paciente = pp.montar_paciente(paciente)
    # 2. Preencher atributos desconhecidos com a moda do treinamento
    nova_paciente = pp.imputar_ausentes(nova_paciente, metadados['modas_para_valores_ausentes'])
    # 3. Codificar os atributos categóricos (mesma função do treinamento)
    nova_paciente = pp.codificar(nova_paciente)[metadados['colunas_codificadas']]
    # 4. Normalizar com o normalizador salvo no treinamento
    nova_paciente = pd.DataFrame(normalizador.transform(nova_paciente),
                                 columns=nova_paciente.columns)
    # 5. Classificar
    classe = modelo.predict(nova_paciente)[0]
    probabilidades = None
    if hasattr(modelo, 'predict_proba'):
        probabilidades = dict(zip(modelo.classes_, modelo.predict_proba(nova_paciente)[0]))
    return classe, probabilidades


def ler_argumentos():
    parser = argparse.ArgumentParser(
        description='Indicativo de diagnóstico (Class) para uma nova paciente - Breast Cancer UCI')
    for atributo in pp.ATRIBUTOS:
        parser.add_argument(f'--{atributo}', dest=atributo,
                            help='Valores aceitos: ' + ', '.join(pp.VALORES_VALIDOS[atributo]) + ', ?')
    argumentos = vars(parser.parse_args())
    if all(valor is None for valor in argumentos.values()):
        print('Nenhum dado informado: usando a paciente de exemplo.\n')
        return NOVA_PACIENTE_EXEMPLO
    faltantes = [a for a, v in argumentos.items() if v is None]
    if faltantes:
        parser.error('informe todos os atributos (use ? para desconhecido). Faltando: '
                     + ', '.join('--' + a for a in faltantes))
    return argumentos


if __name__ == '__main__':
    nova_paciente = ler_argumentos()
    try:
        classe, probabilidades = diagnosticar(nova_paciente)
    except ValueError as erro:
        raise SystemExit(str(erro))
    except FileNotFoundError:
        raise SystemExit('Modelos não encontrados. Execute primeiro: python treinamento.py')

    _, _, metadados = carregar_modelos()
    print('Nova paciente:')
    for atributo in pp.ATRIBUTOS:
        print(f'  {atributo:<12} {nova_paciente[atributo]}')
    print(f'\nModelo: {metadados["algoritmo"]} | acurácia estimada (validação cruzada): '
          f'{metadados["acuracia_revalidacao_cruzada"]:.2%} | acurácia no teste: '
          f'{metadados["acuracia_teste"]:.2%}')
    print(f'Class prevista: {classe}  ->  {DESCRICAO_CLASSE[classe]}')
    if probabilidades:
        for nome_classe, probabilidade in probabilidades.items():
            print(f'  P({nome_classe}) = {probabilidade:.2%}')
    print('\nAviso: resultado de um modelo acadêmico, não substitui avaliação médica.')
