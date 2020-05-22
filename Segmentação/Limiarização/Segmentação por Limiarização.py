def transformar(imagem):
    for i in range(imagem.shape[1]):
        for j in range(imagem.shape[1]):
            if imagem[i][j]==0:
                imagem[i][j]=0
            elif imagem[i][j]==255:
                imagem[i][j]=1
    return imagem 

import os
import math
import numpy as np
import networkx as nx
from skimage import io
from skimage.transform import resize
#from scipy.misc import imsave
def transformar1(imagem):
    
    for i in range(imagem.shape[1]):
        for j in range(imagem.shape[1]):
            if imagem[i][j]==0:
                imagem[i][j]=0
            elif imagem[i][j]==110:
                imagem[i][j]=255
    return imagem
    
def segmentacao_limiar():
    amostra = 0
    raiz =".../Ruido_Impulsivo/"    
    for pasta in os.listdir(raiz):
        arquivos = os.listdir(raiz + pasta)
        for arquivo in arquivos:
            sementes = cv2.imread(raiz + pasta + "/" + arquivo)
            imagem = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY) 
            ret,thresh1 = cv2.threshold(im,30,110,cv2.THRESH_BINARY)
            imagem_mascara = transformar1(thresh1) 
            cv2.imwrite(".../Limiar/mascara/"+str(pasta)+"/"+str(arquivo) + ".bmp", imagem_mascara)
            imagem_binaria = transformar(thresh1)
            imagem_resultado= imagem * imagem_binaria
            cv2.imwrite(".../Limiar/conjuncao/"+str(pasta)+"/"+str(arquivo) + ".bmp", imagem_resultado)
            resultado_ssim = ssim(imagem,imagem_resultado,multichannel=True)
            with open(".../Segmentacao/Limiar/Resultados_Metricas/"+str(pasta)+"/ssim/"+str(arquivo)+".txt","w") as arq2:
                    arq2.write(str(resultado_ssim)) 
            amostra += 1
    return amostra