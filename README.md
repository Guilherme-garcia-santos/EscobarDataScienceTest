
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



Modelo acadêmico, treinado com 286 pacientes de 1988. Não deve ser usado para decisão clínica.
