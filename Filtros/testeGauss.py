# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 21:00:42 2020

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
from scipy import ndimage
from scipy import signal
import math
import numpy.fft as fp
import scipy.fftpack as fp
import scipy.signal
import scipy.ndimage
from skimage.restoration import denoise_nl_means, estimate_sigma
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
            print(file)
            
        
            print('Non local na imagem original') 
            salEpimenta1 =sp_noise(sementes.astype(np.uint8), 0.05) 
            gaussino=cv2.GaussianBlur(sementes,(5,5),0) 
            gaussino=cv2.bitwise_and(gaussino,gaussino,mask=img1)
            gaussino = cv2.cvtColor(gaussino, cv2.COLOR_BGR2GRAY)
            
            sementes1=cv2.bitwise_and(sementes,sementes,mask=img1)           
            imgROI1 = cv2.cvtColor(sementes1, cv2.COLOR_BGR2GRAY)

          
            imgROI = imgROI1[550:3400,550:3400]
            gaussino = gaussino[550:3400,550:3400]
            plt.imshow(gaussino,'gray')
            plt.title("filtro")
            plt.show()
            plt.imshow(imgROI,'gray')
            plt.title("imagem")
            plt.show()
            cv2.imwrite("E:/Dissertacao/Imagens/Gaussiano_Imagem_Original/"+str(folder)+"/"+str(file) + ".bmp", gaussino)
            
            d = psnr4(imgROI.astype(np.uint8), gaussino.astype(np.uint8))
            d1 = mse4(imgROI.astype(np.uint8), gaussino.astype(np.uint8))
            d2 = ssim4(imgROI.astype(np.uint8), gaussino.astype(np.uint8),multichannel=True)
            
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Gaussiano/'+str(folder)+'/psnr/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d))
            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Gaussiano/'+str(folder)+'/mse/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d1))
            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Gaussiano/'+str(folder)+'/ssim/'+str(folder)+str(file)+'.txt','w') as arq2:
                arq2.write(str(d2))
            del(arq2)  
           
            
            print('imagem com ruido sal e pimenta com add 0.05')
            gaussino_ruido=cv2.GaussianBlur(salEpimenta1,(5,5),0)             
            gaussino_ruido=cv2.bitwise_and(gaussino_ruido,gaussino_ruido,mask=img1)
            gaussino_ruido = cv2.cvtColor(gaussino_ruido, cv2.COLOR_BGR2GRAY)     
            
            gaussino_ruido = gaussino_ruido[550:3400,550:3400]
            plt.imshow(gaussino_ruido,'gray')
            plt.title('non local')
            plt.show()
            
            plt.imshow(imgROI,'gray')
            plt.title('non local sem ruido')
            plt.show()
#            
            
            print('Non local na imagem ruidosa')  
            
            cv2.imwrite("E:/Dissertacao/Imagens/Gaussiano_Imagem_ruido/"+str(folder)+"/"+str(file) + ".bmp", gaussino_ruido)
                               
            d = psnr4(imgROI.astype(np.uint8), gaussino_ruido.astype(np.uint8))
            d1 = mse4(imgROI.astype(np.uint8), gaussino_ruido.astype(np.uint8))
            d2 = ssim4(imgROI.astype(np.uint8), gaussino_ruido.astype(np.uint8),multichannel=True)
            
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Gaussiano_ruido/'+str(folder)+'/psnr/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d))
            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Gaussiano_ruido/'+str(folder)+'/mse/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d1))
            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Gaussiano_ruido/'+str(folder)+'/ssim/'+str(folder)+str(file)+'.txt','w') as arq2:
                arq2.write(str(d2))
            del(arq2)  
            ###11111111111111111111111111111111111111111111111111##
            print('f')
            

           
            

            amostra = amostra + 1
                
            soma =soma +1 
    #arq2.close()
    #arq3.close()
    #os.close(comp1)

    return lista


t = ler()

#print(t)
    





cv2.waitKey(0)
cv2.destroyAllWindows()
