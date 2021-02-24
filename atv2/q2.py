import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report,confusion_matrix
from math import sqrt

#Lendo os dados, usamos o mesmo dataset fornecido na primeira questão
dados = pd.read_csv('dados-ex5.txt') 
print(dados.shape)
dados.describe().transpose()

#Coluna Alvo
target_column = ['1.000000000000000000e+00'] 
predicao = list(set(list(dados.columns))-set(target_column))
dados[predicao] = dados[predicao]/dados[predicao].max()
dados.describe().transpose()

X = dados[predicao].values
y = dados[target_column].values

X_treino, X_test, y_treino, y_test = train_test_split(X, y, test_size=0.30, random_state=40)

print(X_treino.shape)
print(X_test.shape)

mlp = MLPClassifier(hidden_layer_sizes=(50,50,50,50,50,50,50), max_iter=100, alpha=1e-4, activation='relu',
                    solver='adam', verbose=10, random_state=1,
                    learning_rate_init=.1)

mlp.fit(X_treino,y_treino)
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate =" + str(0.001))
plt.plot(mlp.loss_curve_)
plt.show()


N_TREINO = X_treino.shape[0]
N_EPOCHS = 60
N_BATCH = 128
N_CLASSES = np.unique(y_treino)
scores_treino = []
scores_test = []
epoch = 0

while epoch < N_EPOCHS:
    print('epoch: ', epoch)
    random_perm = np.random.permutation(X_treino.shape[0])
    mini_batch_index = 0
    while True:
        indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
        mlp.partial_fit(X_treino[indices], y_treino[indices], classes=N_CLASSES)
        mini_batch_index += N_BATCH

        if mini_batch_index >= N_TREINO:
            break
            
    # SCORE DO TREINO
    scores_treino.append(mlp.score(X_treino, y_treino))

    # SCORE TESTE
    scores_test.append(mlp.score(X_test, y_test))

    epoch += 1

fig, ax = plt.subplots(2, sharex=True, sharey=True)
ax[0].plot(scores_treino)
ax[0].set_title('Treino')
ax[1].plot(scores_test)
ax[1].set_title('Test')
fig.suptitle("Accuracy over epochs", fontsize=14)
plt.show()