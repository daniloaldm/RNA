X = []
y = []

arq_address = r'dados/teste.dat'
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


print(X,'\n')
print(adiciona_1_matriz(X, 1))
