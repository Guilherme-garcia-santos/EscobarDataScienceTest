# Breast Cancer (UCI): classificação de recorrência

Trabalho de Data Science. Usa a base [Breast Cancer da UCI](https://archive.ics.uci.edu/dataset/14/breast+cancer)
para prever o atributo **Class**, que indica se houve recorrência do câncer de mama:

- `recurrence-events`: indicativo de recorrência
- `no-recurrence-events`: indicativo de não recorrência

O repositório tem os dois módulos pedidos:

| Módulo | O que faz |
|---|---|
| [`treinamento.py`](treinamento.py) | Executa todos os procedimentos vistos em sala, compara 7 algoritmos e salva o modelo de **melhor acurácia** |
| [`inferencia.py`](inferencia.py) | Recebe uma nova paciente e devolve o indicativo de diagnóstico (Class) |

## Estrutura

```
breast-cancer-ml/
├── dados/
│   └── breast-cancer.csv                  base original (286 pacientes, 9 atributos + Class)
├── modelos/
│   ├── normalizador_breast_cancer.model   MinMaxScaler treinado (pickle)
│   ├── breast_cancer_classificador.model  classificador vencedor (pickle)
│   └── metadados.json                     algoritmo, hiperparâmetros, acurácias e modas
├── resultados/
│   ├── comparacao_modelos.csv             acurácia de cada algoritmo
│   └── saida_treinamento.txt              saída completa da última execução do treino
├── preprocessamento.py                    funções usadas pelo treino E pela inferência
├── treinamento.py                         módulo de treinamento
├── inferencia.py                          módulo de inferência
└── requirements.txt
```

## Como executar

```bash
pip install -r requirements.txt
python treinamento.py
python inferencia.py
```

O treinamento leva cerca de 5 minutos. Os modelos já treinados estão em `modelos/`,
então a inferência funciona sem treinar de novo.

Para classificar uma paciente específica, informe os 9 atributos:

```bash
python inferencia.py --age 40-49 --menopause premeno --tumor-size 30-34 --inv-nodes 3-5 --node-caps yes --deg-malig 3 --breast left --breast-quad left_low --irradiat no
```

Um atributo desconhecido pode ser passado como `?`. Ele é preenchido com a moda do treino.
Valores fora do domínio da UCI são rejeitados com a lista de valores aceitos.

Saída da paciente de exemplo:

```
Modelo: DecisionTree | acurácia estimada (validação cruzada): 76.50% | acurácia no teste: 62.79%
Class prevista: recurrence-events  ->  indicativo de RECORRÊNCIA do câncer de mama
  P(no-recurrence-events) = 28.57%
  P(recurrence-events) = 71.43%
```

A função `diagnosticar(paciente)` também pode ser importada em outro código:

```python
from inferencia import diagnosticar
classe, probabilidades = diagnosticar({'age': '50-59', 'menopause': 'ge40', 'tumor-size': '15-19',
    'inv-nodes': '0-2', 'node-caps': 'no', 'deg-malig': '1', 'breast': 'right',
    'breast-quad': 'central', 'irradiat': 'no'})
```

## Base de dados

| Atributo | Valores | Tratamento |
|---|---|---|
| age | faixas de 10 anos (10-19 … 90-99) | ponto médio da faixa |
| menopause | lt40, ge40, premeno | one-hot (3 colunas) |
| tumor-size | faixas de 5 mm (0-4 … 55-59) | ponto médio da faixa |
| inv-nodes | faixas de 3 linfonodos (0-2 … 36-39) | ponto médio da faixa |
| node-caps | yes, no | 1 / 0 |
| deg-malig | 1, 2, 3 | inteiro |
| breast | left, right | 0 / 1 |
| breast-quad | left_up, left_low, right_up, right_low, central | one-hot (5 colunas) |
| irradiat | yes, no | 1 / 0 |
| **Class** | no-recurrence-events (201), recurrence-events (85) | classe a prever |

A base tem 9 valores ausentes marcados com `?`: 8 em node-caps e 1 em breast-quad.
As faixas viram o ponto médio porque são ordenadas: 40-49 é mais velho que 30-39.
Os atributos nominais viram colunas 0/1 para não criar uma ordem que não existe.

## Procedimentos do treinamento

O `treinamento.py` segue a sequência vista em sala, numerada no próprio código:

1. **Carregar os dados** e converter `?` em valor ausente.
2. **Separar atributos e classe.**
3. **Separar treino (70%) e teste (30%)**, estratificado. O teste não participa de nenhuma escolha.
4. **Tratar ausentes** com a moda, calculada só no treino.
5. **Codificar** os atributos categóricos conforme a tabela acima.
6. **Hiperparametrizar** 7 algoritmos com `RandomizedSearchCV`: 40 combinações cada,
   validação cruzada estratificada de 5 partes repetida 3 vezes, métrica acurácia.
   O **balanceamento SMOTE** e a **normalização MinMax** ficam dentro de um pipeline,
   para serem ajustados só com a parte de treino de cada rodada.
   Cada algoritmo é testado **com e sem** o balanceamento.
7. **Escolher o modelo de melhor acurácia.** A melhor configuração de cada algoritmo é revalidada
   em uma validação cruzada nova, com outras divisões e 10 repetições, e vence a maior acurácia.
   O vencedor é então avaliado uma única vez no teste.
8. **Treinamento definitivo** com as 286 pacientes: balancear, se o vencedor usa SMOTE,
   normalizar, salvar o normalizador, treinar e salvar o modelo, como no exemplo da Iris.

Algoritmos comparados: DecisionTree, RandomForest, GradientBoosting, KNN, SVM,
LogisticRegression e NaiveBayes. Também é calculada uma **referência** que sempre
responde a classe majoritária. Um modelo só é útil se superar essa referência.

## Resultados

| Algoritmo | SMOTE | Acurácia na revalidação cruzada | Acurácia no teste |
|---|---|---|---|
| **DecisionTree** | não | **76,5% ± 5,9** | 62,8% |
| RandomForest | não | 76,2% ± 4,4 | 69,8% |
| GradientBoosting | não | 76,1% ± 4,0 | 68,6% |
| KNN | não | 75,6% ± 3,3 | 70,9% |
| SVM | não | 74,5% ± 3,6 | 72,1% |
| DecisionTree | sim | 72,9% ± 5,5 | 68,6% |
| LogisticRegression | não | 72,8% ± 3,7 | 69,8% |
| NaiveBayes | não | 72,5% ± 3,3 | 70,9% |
| RandomForest | sim | 71,0% ± 4,7 | 65,1% |
| SVM | sim | 70,6% ± 5,3 | 68,6% |
| LogisticRegression | sim | 70,5% ± 1,0 | 69,8% |
| GradientBoosting | sim | 70,3% ± 5,7 | 60,5% |
| KNN | sim | 67,9% ± 6,4 | 66,3% |
| NaiveBayes | sim | 58,1% ± 10,0 | 57,0% |
| *Referência: sempre "no-recurrence-events"* | | *70,5%* | *69,8%* |

**Modelo escolhido:** árvore de decisão sem balanceamento, com `criterion=entropy`,
`splitter=random`, `max_depth=5`, `min_samples_split=18` e `min_samples_leaf=6`.

## Discussão

**O balanceamento reduziu a acurácia em todos os algoritmos.** O SMOTE cria pacientes
sintéticas da classe minoritária e desloca o modelo para prever mais recorrência.
Isso aumenta o acerto dos casos de recorrência, mas aumenta mais os erros nos casos sem
recorrência, que são 70% da base. Como o trabalho pede ênfase na acurácia, o balanceamento
foi avaliado como opção, e não aplicado obrigatoriamente. Para aplicar o SMOTE em todos os
modelos, como no roteiro da aula, basta usar `OPCOES_BALANCEAMENTO = [True]` no início do
`treinamento.py`.

**Por que a escolha não usa o teste.** O teste tem só 86 pacientes, e cada paciente vale
1,2 ponto de acurácia. A margem de erro de uma acurácia de 70% nesse tamanho é de cerca de
5 pontos para cada lado. Escolher o modelo pelo teste seria ajustar a escolha ao acaso dessas
86 pacientes, e o teste deixaria de ser uma avaliação independente. Por isso a escolha usa a
revalidação cruzada no treino, e o teste é usado uma única vez, no vencedor. Os quatro
primeiros colocados ficaram a menos de 1 ponto entre si, uma diferença dentro do ruído.

**Esta base é difícil.** Os atributos são faixas largas e a classe depende de fatores que
não estão na base. Na literatura, as acurácias ficam em torno de 70% a 78%, próximas da
referência. O modelo escolhido supera a referência em 6 pontos na validação cruzada.

**Duplicatas.** Há 14 linhas repetidas. Elas foram mantidas porque são pacientes diferentes
com os mesmos atributos, e não erros de digitação.

## Aviso

Modelo acadêmico, treinado com 286 pacientes de 1988. Não deve ser usado para decisão clínica.
