import cv2
import os
from skimage.measure import compare_ssim as ssim
import numpy as np
import math
from skimage import data, io, filters
def limiar (imagem):
    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            if imagem[i][j]==0:
                imagem[i][j]=0
            elif imagem[i][j]==255:
                imagem[i][j]=1
    return imagem 
def limiar5 (imagem):
    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            if imagem[i][j]==0:
                imagem[i][j]=0
            elif imagem[i][j]==1:
                imagem[i][j]=255
    return imagem
    
def segmentacao_otsu():
    amostra = 0
    raiz =".../Ruido_Impulsivo/"   
    for pasta in os.listdir(raiz):
        arquivos = os.listdir(raiz + pasta)
        for arquivo in arquivos:
            sementes = cv2.imread(raiz + pasta + "/" + arquivo)
            imagem = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY) 
            threshold = filters.threshold_otsu(imagem)
            thresh1 = imagem >= threshold
            mascara = np.ones(imagem.shape,np.uint8) #crio uma mascara com valores 1
            imagem_binaria = mascara * thresh1 
            imagem_mascara = limiar5(imagem_binaria)
            imagem_resultado= imagem * thresh1
            cv2.imwrite(".../Otsu/"+str(pasta)+"/"+str(arquivo) + ".bmp", imagem_resultado)
            resultado_ssim = ssim(imagem,imagem_resultado,multichannel=True)
            with open(".../Segmentacao/Otsu/Resultados_Metricas/"+str(pasta)+"/ssim/"+str(arquivo)+".txt","w") as arq2:
                    arq2.write(str(resultado_ssim)) 
            amostra += 1
    return amostra

cv2.waitKey(0)
cv2.destroyAllWindows()