import math
import matplotlib.pyplot as plt

# Aula 1
# funções de ativação -----------------------------------------------------
def sign(alfa):
    if alfa <= 0:
        return -1
    else:
        return 1

def logistic(alfa):
    return 1 / 1 + math.exp(-alfa)

def tanh(alfa):
    return (math.exp(2 * alfa) - 1) / (math.exp(2 * alfa) + 1)

def relu(alfa):
    if alfa <= 0:
        return 0
    else:
        return alfa
# --------------------------------------------------------------------------

# função que calcula o produto interno de dois vetores ---------------------
def inner_product(v1, v2):
    result = 0
    for i in range(len(v1)):
        result += float(v1[i]) * float(v2[i])
    return result
# --------------------------------------------------------------------------

# função que multiplica um elemento por um vetor ---------------------------
def product(e, v):
    result = []
    for i in range(len(v)):
        x = float(e) * float(v[i])
        result.append(x)
    return result
# --------------------------------------------------------------------------

# função que soma dois vetores ---------------------------------------------
def sum(v1, v2):
    result = []
    for i in range(len(v1)):
        x = float(v1[i]) + float(v2[i])
        result.append(x)
    return result
# --------------------------------------------------------------------------

# função que calcula a porcentagem de similaridade entre dois vetores ------
def score(v1, v2):
    hits = 0
    for i in range(len(v1)):
        if(float(v1[i]) == float(v2[i])):
            hits += 1
    return hits/len(v1)
# --------------------------------------------------------------------------

#-----------------------------Manipulando Matrizes--------------------------
def cria_matriz_com_bies(num_linhas, num_colunas, bies):
    matriz = []
    for i in range(num_linhas):
        linha = []
        for j in range(num_colunas):
            linha.append(1)

        matriz.append(linha)
    return matriz

def adiciona_1_matriz(matriz, bies):
    new_mtz = cria_matriz_com_bies(len(matriz), len(matriz[0])+1, bies)
    for i in range(len(new_mtz)):
        for j in range(len(new_mtz[0])-1):
            new_mtz[i][j-2]=matriz[i][j]
    return new_mtz
#---------------------------------------------------------------------------

# coletando os dados de entada X e y ---------------------------------------
X = []
y = []

arq_address = r'dados/data_or.dat'
f = open(arq_address,"r")
row = f.readline().replace('\n','')
while row:
    columns = row.split(" ")
    # guardo todas as colunas, menos a última (predição)
    X.append(columns[:len(columns) - 1])
    # guardo a última coluna (predição)
    y.append(columns[-1])
    row = f.readline().replace('\n', '')
f.close()

X = adiciona_1_matriz(X, 1)
print(X[0])
# --------------------------------------------------------------------------

#Aula 2
# pesos de conexão ---------------------------------------------------------
W = []
bies = 0
# para cada coluna (feature), adiciono pesos Wi
for i in range(len(X[0])):
    if(i==0):
        W.append(bies) # pode ser randomizado (intervalor de -1 até 1, por exemplo)
    else:
        W.append(0)

# print(W)
# --------------------------------------------------------------------------

# efetuando o treinamento do Perceptron ------------------------------------
# predições treinadas
y_train = []
# número de iterações t do treinamento
T = 5
for t in range(T):
    # guardara as predições em treinamento
    y_training = []

    # percorre todas as predições (vetor y)
    for n in range(len(y)):
        # calculando a predição do perceptron aplicando a função de ativação
        yn = sign(inner_product(W, X[n]))

        # averiguando se houve erro de classificação para aplicar a correção
        if y[n] != yn:
            W = sum(W, product(y[n], X[n]))

        y_training.append(yn)
    
    # atualizando as predições
    y_train = y_training
# --------------------------------------------------------------------------

#Aula 3
# plotagem da superfície de decisão (dados + hiperplano) -------------------
# transferindo toda coluna (feature) para um array

exemples_in_column = []
for j in range(len(X[0])):
    ex = []
    for i in range(len(X)):
        ex.append(X[i][j])
    exemples_in_column.append(ex)

plt.scatter(exemples_in_column[1], exemples_in_column[2], c = y_train)

# plt.scatter(exemples_in_column[0], y_train)
# plt.scatter(exemples_in_column[1], y_train)
# plt.plot(W, [0,0])

plt.show()
# --------------------------------------------------------------------------

# acerto(%) das predições treinadas em relação as reais usadas para o treino
print("score:", score(y_train, y))
# --------------------------------------------------------------------------

#Percepton é feito para classificar dados linearmente divisíveis, notamos que o arquivo data_or.dat 
#respeita essa característica, já o data_xor.dat não. Ao plotar o gráfico fica visível essa anotação.

# https://youtu.be/oud8BZ4ZEac?list=PLh7r2qrOFSigm_MHanQIm4tiAojj1OUvR&t=364
# https://pt.wikipedia.org/wiki/Perceptron_multicamadas#Perceptron