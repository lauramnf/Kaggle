import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.linear_model import LogisticRegression

# Mostrando arquivos
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Abrindo arquivo de treino no modo leitura do módulo pandas
arquivo_treino = pd.read_csv("/kaggle/input/titanic/train.csv")
arquivo_treino.head()

# Abrindo arquivo de testes no modo leitura do módulo pandas
arquivo_teste = pd.read_csv("/kaggle/input/titanic/test.csv")
arquivo_teste.head()

# Aqui verifica-se quais elementos estão classificados como 'nulos' (falta de dados).
# Dependendo do dado, pode-se substituir os valores faltantes pelo valor que prevalece.
# Porem, para a nossa analise, como não há um valor que prevalece de forma expressiva, vamos substituir por um valor independente
# Verifica-se isso com 'objetos_treino["Cabin"].value_counts()'
arquivo_treino[arquivo_treino.isnull().any(axis=1)]

# Retirnando valores nulos
arquivo_treino = arquivo_treino.fillna({"Cabin": "nao identificado"})
ageNa = arquivo_treino['Age'].mean(skipna = True)
arquivo_treino = arquivo_treino.fillna({'Age': ageNa})

# Usando One Hot enconding para mudar valores
arquivo_atualizado = pd.get_dummies(arquivo_treino, columns=["Name", "Sex", "Ticket", "Cabin", "Embarked"], prefix=["name", "sex", "ticket", "cabin", "embarked"], dtype=np.int64)

# A mesma coisa para o arquivo de teste
arquivo_teste[arquivo_teste.isnull().any(axis=1)]
arquivo_teste = arquivo_teste.fillna({"Cabin": "nao identificado"})
arquivo_teste = arquivo_teste.fillna({'Age': ageNa})
arquivo_teste.head()

y_treino = (arquivo_atualizado['Survived']).values
x_treino = (arquivo_atualizado.drop('Survived', axis=1)).values

arquivo_atualizado.head()

arquivo_atualizado.dtypes
arquivo_atualizado.isnull().sum()

molde = LogisticRegression()
molde.fit(x_treino,y_treino)
y_teste = molde.predict(x_teste)
saida = pd.DataFrame({'PassengerId': teste_atualizado.PassengerId, 'Survived': y_teste})
saida.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
