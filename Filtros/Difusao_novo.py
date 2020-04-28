# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 21:00:42 2020

@author: andre
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 21:22:18 2019

@author: andre
"""

import cv2
import os
import re
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
            print(file)
            
        
            print('Difusao na imagem original') 
            salEpimenta1 =sp_noise(sementes.astype(np.uint8), 0.05) 
            difusao=difusao_anisotropica(sementes)
            difusao=cv2.bitwise_and(difusao,difusao,mask=img1)
            difusao = cv2.cvtColor(difusao, cv2.COLOR_BGR2GRAY)
            
            sementes1=cv2.bitwise_and(sementes,sementes,mask=img1)           
            imgROI1 = cv2.cvtColor(sementes1, cv2.COLOR_BGR2GRAY)
            
            #dst = cv2.cvtColor(gaussino, cv2.COLOR_BGR2GRAY)
            imgROI = imgROI1[550:3400,550:3400]
            difusao = difusao[550:3400,550:3400]
            plt.imshow(difusao,'gray')
            plt.title("a")
            plt.show()
            
            # cv2.imwrite("E:/Dissertacao/Imagens/Difusao_Imagem_Original/"+str(folder)+"/"+str(file) + ".bmp", difusao)
            d1 = np.mean((imgROI - difusao) ** 2, dtype=np.float64)
            d = 10 * np.log10((255 ** 2) / d1)
            d2 = ssim4(imgROI.astype(np.uint8), difusao.astype(np.uint8),multichannel=True)
            
            with open('E:/Dissertacao/Resultados_Metricas/Filtro1/Difusao/'+str(folder)+'/psnr/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d))
            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro1/Difusao/'+str(folder)+'/mse/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d1))
            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro1/Difusao/'+str(folder)+'/ssim/'+str(folder)+str(file)+'.txt','w') as arq2:
                arq2.write(str(d2))
            del(arq2)  
           
            
            print('imagem com ruido sal e pimenta com add 0.05')
            difusao_ruido=difusao_anisotropica(salEpimenta1)            
            difusao_ruido=cv2.bitwise_and(difusao_ruido,difusao_ruido,mask=img1)
            difusao_ruido = cv2.cvtColor(difusao_ruido, cv2.COLOR_BGR2GRAY)     
            
            difusao_ruido = difusao_ruido[550:3400,550:3400]
            
            
            
            salEpimenta1=cv2.bitwise_and(salEpimenta1,salEpimenta1,mask=img1)
            salEpimenta1 = cv2.cvtColor(salEpimenta1, cv2.COLOR_BGR2GRAY)
            # cv2.imwrite("E:/Dissertacao/imagens_colocar_dissertacao/ruido_salepimenta/"+str(folder)+"/"+str(file) + ".jpeg", salEpimenta1)
            
            salEpimenta1 = salEpimenta1[550:3400,550:3400]
            plt.imshow(salEpimenta1,'gray')
            plt.title(' ruido')
            plt.show()
            
            plt.imshow(difusao_ruido,'gray')
            plt.title('gaussiano ruido')
            plt.show()
            
            plt.imshow(imgROI,'gray')
            plt.title('gaussiano sem ruido')
            plt.show()
            
            # cv2.imwrite("E:/Dissertacao/Imagens/Difusao_Imagem_Ruido/"+str(folder)+"/"+str(file) + ".bmp", difusao_ruido)
            print('Difusao na imagem ruidosa')   
            
                    
            d1 = np.mean((imgROI - difusao_ruido) ** 2, dtype=np.float64)
            d = 10 * np.log10((255 ** 2) / d1)
            d2 = ssim4(imgROI.astype(np.uint8), difusao_ruido.astype(np.uint8),multichannel=True)
            
            with open('E:/Dissertacao/Resultados_Metricas/Filtro1/Difusao_ruido/'+str(folder)+'/psnr/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d))
            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro1/Difusao_ruido/'+str(folder)+'/mse/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d1))
            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro1/Difusao_ruido/'+str(folder)+'/ssim/'+str(folder)+str(file)+'.txt','w') as arq2:
                arq2.write(str(d2))
            del(arq2)  
            ###11111111111111111111111111111111111111111111111111##
            
            
               
           
            

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
