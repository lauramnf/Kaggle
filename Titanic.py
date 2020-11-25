import numpy as np # linear algebra
import pandas as pd # data processing
import os #interact with operational system
from sklearn.linear_model import LogisticRegression #machine learning model

# Showing files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Oppening training files
arquivo_treino = pd.read_csv("/kaggle/input/titanic/train.csv")

# Openning testing file
arquivo_teste = pd.read_csv("/kaggle/input/titanic/test.csv")

# Verify missing data in training and testing files (null elements)
arquivo_treino[arquivo_treino.isnull().any(axis=1)]
arquivo_teste[arquivo_teste.isnull().any(axis=1)]

# Fill the missing in training and testing data
arquivo_treino = arquivo_treino.fillna({"Cabin": "nao identificado"})
ageNa = arquivo_treino['Age'].mean(skipna = True)
arquivo_treino = arquivo_treino.fillna({'Age': ageNa})

arquivo_teste = arquivo_teste.fillna({"Cabin": "nao identificado"})
arquivo_teste = arquivo_teste.fillna({'Age': ageNa})

# Use One Hot Enconding module to change categorical data into numerical data in both files
arquivo_atualizado = pd.get_dummies(arquivo_treino, columns=["Name", "Sex", "Ticket", "Cabin", "Embarked"], prefix=["name", "sex", "ticket", "cabin", "embarked"], dtype=np.int64)
teste_atualizado = pd.get_dummies(arquivo_teste, columns=["Name", "Sex", "Ticket", "Cabin", "Embarked"], prefix=["name", "sex", "ticket", "cabin", "embarked"], dtype=np.int64)


# Get the testing data and the corresponding result separately
y_treino = (arquivo_atualizado['Survived']).values
x_treino = (arquivo_atualizado.drop('Survived', axis=1)).values

# Get the testing data separately
x_teste = teste_atualizado.values

# Create a object LogisticRegression
molde = LogisticRegression()
molde.fit(x_treino,y_treino) #Fiting training data into the module
y_teste = molde.predict(x_teste) # Predict testing data results
saida = pd.DataFrame({'PassengerId': teste_atualizado.PassengerId, 'Survived': y_teste})
saida.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
