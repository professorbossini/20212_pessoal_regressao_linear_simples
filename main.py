import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv ('dados_regressao_linear_simples.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_treinamento, y_treinamento, y_treinamento, y_teste = (
    train_test_split(x, y, test_size=0.2, random_state=0)
)

# print (x_treinamento, y_treinamento, y_treinamento, y_teste)

linearRegression = LinearRegression()

linearRegression.fit(x_treinamento, y_treinamento)

y_pred = linearRegression.predict(x_treinamento)

for y_1, y_2 in zip (y_treinamento, y_pred.reshape(len(y_pred), 1)):
    print (y_1, y_2[0])

plt.scatter(x_treinamento, y_treinamento, color='red')
plt.plot(x_treinamento, y_pred, color='blue')
plt.title('Salário vs Tempo de experiência (treinamento)')
plt.xlabel('Anos de Experiência')
plt.ylabel('Salário')
plt.show()
