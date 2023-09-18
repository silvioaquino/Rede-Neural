# pip install keras
# https://didatica.tech/wp-content/uploads/2023/02/admission_dataset.csv

import keras

import pandas as pd 
import warnings 
warnings.filterwarnings('ignore')
# https://didatica.tech/wp-content/uploads/2023/02/admission_dataset.csv
df = pd.read_csv('~/Mesa/Programação Studio/Python/VSCode/Rede_Neural/arquivos/admission_dataset.csv')

df 

y = df['Chance of Admit ']
x = df.drop('Chance of Admit ', axis = 1)

x_treino, x_teste = x[0:300], x[300:]
y_treino, y_teste = y[0:300], y[300:]

x_treino.shape

from keras.models import Sequential
from keras.layers import Dense

# Criando a arquitetura da rede neural:

modelo = Sequential()
modelo.add(Dense(units=3, activation='relu', input_dim=x_treino.shape[1]))
modelo.add(Dense(units=1, activation='linear'))

# Treinando a rede neural:
modelo.compile(loss='mse', optimizer='adam', metrics=['mae'])
resultado = modelo.fit(x_treino, y_treino, epochs=200, batch_size=32, validation_data=(x_teste, y_teste))

import matplotlib.pyplot as plt

#plotar grafico do historico de treinamento
plt.plot(resultado.history['loss'])
plt.plot(resultado.history['val_loss'])
plt.title('Histórico de Treinamento')
plt.ylabel('Função de custo')
plt.xlabel('Épocas de treinamento')
plt.legend(['Erro treino', 'Erro teste'])
plt.show()



