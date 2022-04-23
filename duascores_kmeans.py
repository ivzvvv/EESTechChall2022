from matplotlib import image as img
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import whiten
import pandas as pd
from scipy.cluster.vq import kmeans
import random
import math as m

def hilo(a, b, c):
    if c < b: b, c = c, b
    if b < a: a, b = b, a
    if c < b: b, c = c, b
    return a+c

def complement(r, g, b):
    k = hilo(r, g, b)
    return tuple(k - u for u in (r, g, b))

f = open('modelo_treinado.csv', 'w')
f_t = open('train.csv', 'r')

data_f = pd.read_csv('train.csv')
data_f.columns
palettes = list()
image_names = data_f.image
for i in range(0, 10):
    palettes.append(list())
    image = img.imread('./dataset/dataset/'+ str(image_names[random.randint(0, len(image_names))]) +'.jpg')
    image.shape

    #fig, ax = plt.subplots(nrows = 1, ncols=1, figsize=(15,5))

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



    cluster_centers, distortion = kmeans(df[['r_whiten', 'g_whiten', 'b_whiten']], 2)
    if len(cluster_centers) < 2:
        i-=1
        continue

    r_std, g_std, b_std = df[['r', 'g', 'b']].std()
    
    
    colors=[]
    sr, sg, sb = cluster_centers[0]
    if  int(sr*r_std) < 50 and  int(sg*g_std) < 50 and int(sb*b_std) < 50:
        #primary darker
        colors.append((int(sr*r_std), int(sg*g_std), int(sb*b_std)))
        #primary 
        colors.append((int(sr*r_std*2), int(sg*g_std*2), int(sb*b_std*2)))
        #primary brighter
        colors.append((min(255, int(sr*r_std*2*1.5)), min(255, int(sg*g_std*2*1.5)), min(255, int(sb*b_std*2*1.5))))
    elif int(sr*r_std) > 200 and  int(sg*g_std) > 200 and int(sb*b_std) > 200:
        #primary darker
        colors.append((int(sr*r_std/1.25/1.25), int(sg*g_std/1.25/1.25), int(sb*b_std/1.25/1.25)))
        #primary 
        colors.append((int(sr*r_std/1.25), int(sg*g_std/1.25), int(sb*b_std/1.25)))
        #primary brighter
        colors.append((int(sr*r_std), int(sg*g_std), int(sb*b_std)))
        
    elif int(sr*r_std) < 70 and  int(sg*g_std) < 70 and int(sb*b_std) < 70:
        #primary darker
        colors.append((int(sr*r_std), int(sg*g_std), int(sb*b_std)))
        #primary 
        colors.append((int(sr*r_std*1.5), int(sg*g_std*1.5), int(sb*b_std*1.5)))
        #primary brighter
        colors.append((min(255, int(sr*r_std*1.5*1.5)), min(255, int(sg*g_std*1.5*1.5)), min(255, int(sb*b_std*1.5*1.5))))
    else:
        #primary darker
        colors.append((int(sr*r_std/1.25), int(sg*g_std/1.25), int(sb*b_std/1.25)))
        #primary 
        colors.append((int(sr*r_std), int(sg*g_std), int(sb*b_std)))
        #primary brighter
        colors.append((min(255, int(sr*r_std*1.25)), min(255, int(sg*g_std*1.25)), min(255, int(sb*b_std*1.25))))
    
    sr, sg, sb = cluster_centers[1]
    if  int(sr*r_std) < 50 and  int(sg*g_std) < 50 and int(sb*b_std) < 50:
        #primary darker
        colors.append((int(sr*r_std), int(sg*g_std), int(sb*b_std)))
        #primary 
        colors.append((int(sr*r_std*2), int(sg*g_std*2), int(sb*b_std*2)))
        #primary brighter
        colors.append((min(255, int(sr*r_std*2*1.5)), min(255, int(sg*g_std*2*1.5)), min(255, int(sb*b_std*2*1.5))))
    elif int(sr*r_std) > 200 and  int(sg*g_std) > 200 and int(sb*b_std) > 200:
        #primary darker
        colors.append((int(sr*r_std/1.25/1.25), int(sg*g_std/1.25/1.25), int(sb*b_std/1.25/1.25)))
        #primary 
        colors.append((int(sr*r_std/1.25), int(sg*g_std/1.25), int(sb*b_std/1.25)))
        #primary brighter
        colors.append((int(sr*r_std), int(sg*g_std), int(sb*b_std)))
        
    elif int(sr*r_std) < 70 and  int(sg*g_std) < 70 and int(sb*b_std) < 70:
        #primary darker
        colors.append((int(sr*r_std), int(sg*g_std), int(sb*b_std)))
        #primary 
        colors.append((int(sr*r_std*1.5), int(sg*g_std*1.5), int(sb*b_std*1.5)))
        #primary brighter
        colors.append((min(255, int(sr*r_std*1.5*1.5)), min(255, int(sg*g_std*1.5*1.5)), min(255, int(sb*b_std*1.5*1.5))))
    else:
        #primary darker
        colors.append((int(sr*r_std/1.25), int(sg*g_std/1.25), int(sb*b_std/1.25)))
        #primary 
        colors.append((int(sr*r_std), int(sg*g_std), int(sb*b_std)))
        #primary brighter
        colors.append((min(255, int(sr*r_std*1.25)), min(255, int(sg*g_std*1.25)), min(255, int(sb*b_std*1.25))))

    
    #depois de fazer os clusters, ver o mean da palette e classificar com uma qualidade
    #depois usar algum metodo de regressao linear para escolher palette de acordo com a qualidade pretendida
    #colors.append((int(sum(colors[0:7][0])/7), int(sum(colors[0:7][1])/7), int(sum(colors[0:7][2])/7)))
    
    colors.append(complement(colors[1][0], colors[1][1], colors[1][2]))
    palettes[i].append(colors)
    #plt.imshow([colors])
    print(f'Iteracao {i}')
    f.writelines(str(colors) + ';' + str(data_f.label[i])+ '\n')
    #plt.show()

f.close()


cor_primaria_escolhida = (102, 75, 200)
dist_min = 1000
for p in palettes:
    #               R          G            B
    print(p)
    if m.sqrt((p[0][1][0]-cor_primaria_escolhida[0])**2 + 
              (p[0][1][1]-cor_primaria_escolhida[1])**2 + 
              (p[0][1][2]-cor_primaria_escolhida[2])**2) < dist_min:
        dist_min = m.sqrt((p[0][1][0]-cor_primaria_escolhida[0])**2 + (p[0][1][1]-cor_primaria_escolhida[1])**2 + (p[0][1][2]-cor_primaria_escolhida[2])**2)
        palette_escolhida = p[0]

plt.imshow([palette_escolhida])
plt.show()
