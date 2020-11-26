import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.linear_model import LogisticRegression #machine learning model

def verify_accuracy(submission_file, answer_file):
# Function to verify model's accuracy
    # Read my model submission
    with open(submission_file, 'r') as my_sub:
        my_lines = my_sub.readlines()

    # Read answer file
    with open(answer_file, 'r') as certo_sub:
        certo_lines = certo_sub.readlines()

    # Verify model's accuracy
    diff = 0
    for my_line, certo_line in zip(my_lines, certo_lines):
        if my_line != certo_line: # Check compatible lines
            diff += 1

    perc = 100*(1-diff/len(my_lines))
    print('Your model\'s has accuracy of: ', end = '')
    return f'{perc:.2f}%'

# Oppening training files
arquivo_treino = pd.read_csv("train.csv")
# Openning testing file
arquivo_teste = pd.read_csv("test.csv")

passenger_id = arquivo_teste['PassengerId'] #Saving Passenger ID
# Delete useless data
arquivo_treino = arquivo_treino.drop(columns=['Name', 'Cabin', 'PassengerId', 'Ticket'])
arquivo_teste = arquivo_teste.drop(columns=['Name', 'Cabin', 'PassengerId', 'Ticket'])

# Verify missing data in files (null elements)
# Training data
arquivo_treino[arquivo_treino.isnull().any(axis=1)]
# Testing data
arquivo_teste[arquivo_teste.isnull().any(axis=1)]

# Fill the missing data
# Training data
ageNa = arquivo_treino['Age'].mean(skipna = True)
arquivo_treino = arquivo_treino.fillna({'Fare': 0})
arquivo_treino = arquivo_treino.fillna({'Age': ageNa})
# Testing data
arquivo_teste = arquivo_teste.fillna({'Fare': 0})
arquivo_teste = arquivo_teste.fillna({'Age': ageNa})

# Use One Hot Enconding module to change categorical data into numerical data in both files
teste_atualizado = pd.get_dummies(arquivo_teste, columns=["Sex","Embarked"], prefix=["sex", "embarked"], dtype=np.int64)
arquivo_atualizado = pd.get_dummies(arquivo_treino, columns=["Sex", "Embarked"], prefix=["sex", "embarked"], dtype=np.float64)

# Get the testing data and the corresponding result separately
y_treino = (arquivo_atualizado['Survived']).values
x_treino = (arquivo_atualizado.drop('Survived', axis=1)).values

# Get the testing data separately
x_teste = teste_atualizado.values

# Create a object LogisticRegression
molde = LogisticRegression()
molde.fit(x_treino,y_treino) #Fiting training data into the module
y_teste = molde.predict(x_teste) # Predict testing data results
saida = pd.DataFrame({'PassengerId': passenger_id, 'Survived': y_teste})
saida.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

print(verify_accuracy('my_submission.csv','certo.csv'))
