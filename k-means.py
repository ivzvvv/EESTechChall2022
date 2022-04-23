from matplotlib import image as img
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import whiten
import pandas as pd
from scipy.cluster.vq import kmeans
import random

f = open('modelo_treinado.csv', 'w')
f_t = open('train.csv', 'r')

data_f = pd.read_csv('train.csv')
data_f.columns

image_names = data_f.image
for i in range(0, 10):
    image = img.imread('./dataset/dataset/'+ str(image_names[random.randint(0, len(image_names))]) +'.jpg')
    image.shape

    fig, ax = plt.subplots(nrows = 1, ncols=1, figsize=(15,5))

    """for c, ax in zip(range(3), ax):
        # initiate a zero matrix with dtype as unit8 as the R,G,B values are between 0 to 255
        channel = np.zeros(image.shape, dtype="uint8")
            
        # only allow one color at each time
        channel[:, :, c] = image[:, :, c]
            
        # display each channel
        ax.imshow(channel)

    plt.show()"""

    #construct to a dataframe for future data process
    df = pd.DataFrame()
    df['r']=pd.Series(image[:,:,0].flatten())
    df['g']=pd.Series(image[:,:,1].flatten())
    df['b']=pd.Series(image[:,:,2].flatten())
    df.head()

    df['r_whiten'] = whiten(df['r'])
    df['g_whiten'] = whiten(df['g'])
    df['b_whiten'] = whiten(df['b'])
    df.head()



    cluster_centers, distortion = kmeans(df[['r_whiten', 'g_whiten', 'b_whiten']], 7)
    if len(cluster_centers) < 7:
        i-=1
        continue

    r_std, g_std, b_std = df[['r', 'g', 'b']].std()
    colors=[]

    for color in cluster_centers:
        sr, sg, sb = color
        colors.append((int(sr*r_std), int(sg*g_std), int(sb*b_std)))
        #plt.imshow([colors])

    #depois de fazer os clusters, ver o mean da palette e classificar com uma qualidade
    #depois usar algum metodo de regressao linear para escolher palette de acordo com a qualidade pretendida
    colors.append((int(sum(colors[0:7][0])/7), int(sum(colors[0:7][1])/7), int(sum(colors[0:7][2])/7)))
    plt.imshow([colors])

    print(f'Iteracao {i}')
    f.writelines(str(colors) + ';' + str(data_f.label[i])+ '\n')
    #plt.show()

f.close()
