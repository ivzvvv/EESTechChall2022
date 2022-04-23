from distutils import core
import pandas as pd
import matplotlib.pyplot as plt
import math as m
#f = open('modelo_treinado.csv')

dados = pd.read_csv('modelo_treinado2.csv')
dados.columns
cor_primaria_escolhida = (50, 209, 111)
dist_min = 1000
print(dados)
cores_primarias = list()
for i in range(0, 100):
    aux = []
    aux.append(dados.iloc[i][3])
    aux.append(dados.iloc[i][4])
    aux.append(dados.iloc[i][5])
    cores_primarias.append((aux[0], aux[1], aux[2]))


print(cores_primarias)
ind = 0
cont = 0
cor_escolhida = list()
for cor in cores_primarias:
    #               R          G            B
    if m.sqrt((cor[0]-cor_primaria_escolhida[0])**2 + 
              (cor[1]-cor_primaria_escolhida[1])**2 + 
              (cor[2]-cor_primaria_escolhida[2])**2) < dist_min:
        dist_min = m.sqrt((cor[0]-cor_primaria_escolhida[0])**2 + (cor[1]-cor_primaria_escolhida[1])**2 + (cor[2]-cor_primaria_escolhida[2])**2)
        cor_escolhida = cor
        ind = cont
    cont += 1

palette = list()
for i in range(0, 21, 3):
    if i == 0:
        palette.append((int(cor_primaria_escolhida[0]/1.25), int(cor_primaria_escolhida[1]/1.25), int(cor_primaria_escolhida[2]/1.25)))
    elif i == 3:
        palette.append((int(cor_primaria_escolhida[0]), int(cor_primaria_escolhida[1]), int(cor_primaria_escolhida[2])))
    elif i == 6:
        palette.append((min(255, (int(cor_primaria_escolhida[0]*1.25))), min(255, int(cor_primaria_escolhida[1]*1.25)), min(255, int(cor_primaria_escolhida[2]*1.25))))
    else:
        palette.append((dados.iloc[ind][i], dados.iloc[ind][i+1], dados.iloc[ind][i+2]))
    plt.imshow([palette])
plt.show()