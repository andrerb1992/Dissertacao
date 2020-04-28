# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 21:22:18 2019

@author: andre
"""

import cv2
import os

from matplotlib import pyplot as plt
import numpy as np
import glob
import random
from scipy.fftpack import fft2, ifft2, fftfreq, fftshift
import matplotlib.pylab as pylab
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy import ndimage
from scipy import signal
import math
import numpy.fft as fp
import scipy.fftpack as fp
import scipy.signal
import scipy.ndimage

import sys
from skimage.restoration import denoise_nl_means
from skimage.measure import compare_psnr as psnr4
from skimage.measure import compare_mse as mse4
from skimage.measure import compare_ssim as ssim4
import timeit
import time 
import scipy.fftpack as fftpack
import csv

def sp_noise(image, prob):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def filter_nonlocal_means(img):
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,7,21)
    return dst

def filter_nonlocal_means1(img):
    #sigma = 0.08
    sigma_est = np.mean(estimate_sigma(img, multichannel=True))
    patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=True)

    denoise2 = denoise_nl_means(img, h=0.8 * sigma_est, sigma=sigma_est,
                                fast_mode=False, **patch_kw)

    return denoise2

def difusao_anisotropica(imagem):
    dst =cv2.ximgproc.anisotropicDiffusion(imagem,0.15, 90, 40)
    return dst


def ler():
    #raiz = 'C:/Users/andre/leitura_teste/'

    raiz ='E:/Dissertacao/Imagens/imagem original/'
    #raiz ='C:/Users/andre/OneDrive/Documentos/Base_de_dados_com_100_cada_semente/imagem_ROI1/'
    
    lista = []
    amostra = 1
    id = []
    cont = 0
    soma = 1
    b = ".bmp"
    count = 40
   
    for folder in os.listdir(raiz):
        # print(folder)
        files = os.listdir(raiz + folder)
        print(folder)
        
        id.append(folder + " ")
        
        for file in files:
            # if '.bmp' in file:
            lista.append(raiz + folder + '/' + file)
            sementes = cv2.imread(raiz + folder + '/' + file)            
            img1= np.zeros(sementes.shape[:2],dtype="uint8")
            plt.imshow(img1,'gray')
            plt.show()            
            (cX,cY)=(sementes.shape[1] // 2,sementes.shape[0] // 2)
            cv2.circle(img1,(2000,2000),1395,255,-1)
            img_com_mascara=cv2.bitwise_and(sementes,sementes,mask=img1)
            plt.imshow(img_com_mascara,'gray')
            plt.show()
            imgROI = img_com_mascara[550:3400,550:3400]
            plt.imshow(imgROI,'gray')
            plt.title(folder)
            plt.show()
            print(str(file))     
            
           
            print('Non_Local_Means na imagem original') 
            nonlocal_means=filter_nonlocal_means(imgROI)     
            imgROI1 = cv2.cvtColor(imgROI, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("E:/Dissertacao/Imagens/Non_Local_Means_Imagem_Original/"+str(folder)+"/"+str(file) + ".bmp", nonlocal_means)
            #dst = cv2.cvtColor(nonlocal_means, cv2.COLOR_BGR2GRAY)
            
            d = psnr4(imgROI.astype(np.uint8), nonlocal_means.astype(np.uint8))
            d1 = mse4(imgROI.astype(np.uint8), nonlocal_means.astype(np.uint8))
            d2 = ssim4(imgROI.astype(np.uint8), nonlocal_means.astype(np.uint8),multichannel=True)
          
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Non_local_means/'+str(folder)+'/psnr/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d))

            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Non_local_means/'+str(folder)+'/mse/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d1))

            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Non_local_means/'+str(folder)+'/ssim/'+str(folder)+str(file)+'.txt','w') as arq2:
                arq2.write(str(d2))
            del(arq2)  
            
            print('imagem com ruido sal e pimenta com add 0.05')
            salEpimenta1 =sp_noise(imgROI.astype(np.uint8), 0.05)  
            
            print('Non_Local_Means na imagem ruidosa')            
            nonlocal_means_ruido=filter_nonlocal_means(salEpimenta1)            
            cv2.imwrite("E:/Dissertacao/Imagens/Non_Local_Means_Imagem_ruido/"+str(folder)+"/"+str(file) + ".bmp", nonlocal_means_ruido)
            d = psnr4(imgROI.astype(np.uint8), nonlocal_means_ruido.astype(np.uint8))
            d1 = mse4(imgROI.astype(np.uint8), nonlocal_means_ruido.astype(np.uint8))
            d2 = ssim4(imgROI.astype(np.uint8), nonlocal_means_ruido.astype(np.uint8),multichannel=True)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Non_local_means_ruido/'+str(folder)+'/psnr/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d))
            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Non_local_means_ruido/'+str(folder)+'/mse/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d1))
            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Non_local_means_ruido/'+str(folder)+'/ssim/'+str(folder)+str(file)+'.txt','w') as arq2:
                arq2.write(str(d2))
            del(arq2)  
            
            ###11111111111111111111111111111111111111111111111111##
            del(nonlocal_means)
            del(nonlocal_means_ruido)
            print('Non_Local_Means na imagem original1') 
            nonlocal_means=filter_nonlocal_means1(imgROI)     
            imgROI1 = cv2.cvtColor(imgROI, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("E:/Dissertacao/Imagens/Non_Local_Means_Imagem_Original1/"+str(folder)+"/"+str(file) + ".bmp", nonlocal_means)
            #dst = cv2.cvtColor(nonlocal_means, cv2.COLOR_BGR2GRAY)
            
            d = psnr4(imgROI.astype(np.uint8), nonlocal_means.astype(np.uint8))
            d1 = mse4(imgROI.astype(np.uint8), nonlocal_means.astype(np.uint8))
            d2 = ssim4(imgROI.astype(np.uint8), nonlocal_means.astype(np.uint8),multichannel=True)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Non_local_means1/'+str(folder)+'/psnr/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d))

            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Non_local_means1/'+str(folder)+'/mse/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d1))

            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Non_local_means1/'+str(folder)+'/ssim/'+str(folder)+str(file)+'.txt','w') as arq2:
                arq2.write(str(d2))
            del(arq2)    
            
            print('imagem com ruido sal e pimenta com add 0.05')
            salEpimenta1 =sp_noise(imgROI.astype(np.uint8), 0.05)  
            
            print('Non_Local_Means na imagem ruidosa1')            
            nonlocal_means_ruido=filter_nonlocal_means1(salEpimenta1)            
            cv2.imwrite("E:/Dissertacao/Imagens/Non_Local_Means_Imagem_ruido1/"+str(folder)+"/"+str(file) + ".bmp", nonlocal_means_ruido)
            d = psnr4(imgROI.astype(np.uint8), nonlocal_means_ruido.astype(np.uint8))
            d1 = mse4(imgROI.astype(np.uint8), nonlocal_means_ruido.astype(np.uint8))
            d2 = ssim4(imgROI.astype(np.uint8), nonlocal_means_ruido.astype(np.uint8),multichannel=True)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Non_local_means_ruido1/'+str(folder)+'/psnr/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d))
            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Non_local_means_ruido1/'+str(folder)+'/mse/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d1))
            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Non_local_means_ruido1/'+str(folder)+'/ssim/'+str(folder)+str(file)+'.txt','w') as arq2:
                arq2.write(str(d2))
            del(arq2)  
               
           
            

            amostra = amostra + 1
                
            soma =soma +1 
    #arq2.close()
    #arq3.close()
    #os.close(comp1)

    return lista


t = ler()

print(t)

cv2.waitKey(0)
cv2.destroyAllWindows()
