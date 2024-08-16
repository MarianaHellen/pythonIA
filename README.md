# pythonIA

Projeto desenvolvido durante a jornada Python da escola Hashtag Treinamentos. 
# Credit Score Prediction

Este projeto utiliza Inteligência Artificial para prever o score de crédito de clientes em um banco fictício. O objetivo é classificar os clientes com base em diversos atributos financeiros e demográficos, utilizando dois modelos de aprendizado de máquina: Random Forest e K-Nearest Neighbors (KNN).

## Descrição do Projeto

A previsão do score de crédito é uma tarefa crucial para instituições financeiras, pois ajuda a avaliar o risco associado a conceder empréstimos a diferentes clientes. Este projeto visa construir e avaliar modelos de machine learning capazes de realizar essa previsão com alta precisão.

### Modelos Utilizados

1. **Random Forest Classifier**
   - **Biblioteca**: `sklearn.ensemble`
   - **Descrição**: O Random Forest é um método de aprendizado de máquina baseado em múltiplas árvores de decisão, que combina as previsões de várias árvores para melhorar a precisão e evitar o overfitting.
   - **Acurácia**: 82%

2. **K-Nearest Neighbors (KNN) Classifier**
   - **Biblioteca**: `sklearn.neighbors`
   - **Descrição**: O KNN é um algoritmo simples e eficaz que classifica os dados com base na proximidade dos pontos em um espaço multidimensional. Ele atribui a classe mais comum entre os k vizinhos mais próximos do ponto a ser classificado.
   - **Acurácia**: 73%

### Pré-processamento de Dados

Antes de treinar os modelos, foi necessário realizar o pré-processamento dos dados para garantir que os algoritmos pudessem trabalhar com as informações fornecidas.

1. **Codificação de Rótulos (Label Encoding)**
   - **Biblioteca**: `sklearn.preprocessing`
   - **Descrição**: O `LabelEncoder` foi utilizado para converter atributos categóricos em valores numéricos, facilitando o treinamento dos modelos de machine learning.
   - **Exemplo de Código**:
     ```python
     from sklearn.preprocessing import LabelEncoder
     le = LabelEncoder()
     df['categorical_column'] = le.fit_transform(df['categorical_column'])
     ```

2. **Divisão dos Dados em Conjuntos de Treinamento e Teste**
   - **Biblioteca**: `sklearn.model_selection`
   - **Descrição**: Utilizamos `train_test_split` para dividir o conjunto de dados em conjuntos de treinamento e teste. Isso permite avaliar o desempenho dos modelos em dados que não foram utilizados durante o treinamento.
   - **Exemplo de Código**:
     ```python
     from sklearn.model_selection import train_test_split
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     ```

3. **Manipulação de Dados com Pandas**
   - **Biblioteca**: `pandas`
   - **Descrição**: A biblioteca `pandas` foi utilizada para a manipulação e análise dos dados, permitindo a leitura de arquivos, limpeza e organização dos dados em DataFrames.
   - **Exemplo de Código**:
     ```python
     import pandas as pd
     df = pd.read_csv('data.csv')
     ```

## Tecnologias Utilizadas

- Python 3.x
- Bibliotecas:
  - scikit-learn
  - pandas

## Conclusão

O modelo de Random Forest apresentou uma melhor performance em termos de acurácia (82%) em comparação ao modelo KNN (73%). Esta diferença pode ser atribuída à capacidade do Random Forest de manejar melhor as variáveis e reduzir o risco de overfitting ao combinar múltiplas árvores de decisão.

## Como Executar o Projeto

1. Clone o repositório:
   ```bash
   git clone https://github.com/MarianaHellen/pythonIA.git
