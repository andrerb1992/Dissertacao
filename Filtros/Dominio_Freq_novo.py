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
from skimage.restoration import denoise_nl_means, estimate_sigma
import sys
from skimage.restoration import denoise_nl_means
from skimage.measure import compare_psnr as psnr4
from skimage.measure import compare_mse as mse4
from skimage.measure import compare_ssim as ssim4
from scipy import fftpack
import scipy.fftpack as fftpack
import csv
#from scipy.stats import signaltonoise

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


def filtroFrequenciaAltaIdealLivro(im):
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    freq = fp.fft2(im)
    (w, h) = freq.shape
    half_w, half_h = int(w/2), int(h/2)
    freq1 = np.copy(freq)
    freq2 = fp.fftshift(freq1)
    pylab.figure(figsize=(10,10)), pylab.imshow( (20*np.log10( 0.1 + freq2)).astype(int)), pylab.show()
    freq2[half_w-1:half_w+2,half_h-1:half_h+2] = 0 # select all but the first 20x20 (low) frequencies
    pylab.figure(figsize=(10,10))
    pylab.imshow( (20*np.log10( 0.1 + freq2)).astype(int))
    pylab.show()
    im1 = np.clip(fp.ifft2(fftpack.ifftshift(freq2)).real,0,255) # clip pixel values after IFFT

    pylab.imshow(im1, cmap='gray'), pylab.axis('off'), pylab.show()        
    return im1.astype(np.uint8)

def filtroFrequenciaBaixaIdealLivro(im):
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    freq = fp.fft2(im)
    (w, h) = freq.shape
    half_w, half_h = int(w/2), int(h/2)
    freq1 = np.copy(freq)
    freq2 = fftpack.fftshift(freq1)
    freq2_low = np.copy(freq2)
    freq2_low[half_w-24:half_w+25,half_h-24:half_h+25] = 0 # block the lowfrequencies
    freq2 -= freq2_low # select only the first 20x20 (low) frequencies, block the high frequencies
    im1 = fp.ifft2(fftpack.ifftshift(freq2)).real

    pylab.imshow(im1, cmap='gray'), pylab.axis('off'), pylab.show()        
    return im1.astype(np.uint8)

def filtroFrequenciaBandaIdealLivro(img,img1):
    resultado = img * img1
    plt.imshow(resultado,'gray')
    plt.title('banda')
    plt.show()
    
    return resultado

def ler():
   
    raiz ='E:/Dissertacao/Imagens/imagem original/'
    
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
            print('freq na imagem original') 
            salEpimenta1 =sp_noise(sementes.astype(np.uint8), 0.05) 
            
            sementes1 = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
            gaussinoAlta=filtroFrequenciaAltaIdealLivro(sementes1)
            baixarfeq=filtroFrequenciaBaixaIdealLivro(sementes1)
            
            gaussino=filtroFrequenciaBandaIdealLivro(gaussinoAlta,baixarfeq)
            print(gaussino.shape)          
            gaussino=cv2.bitwise_and(gaussino,gaussino,mask=img1)
            sementes1=cv2.bitwise_and(sementes,sementes,mask=img1)           
            imgROI1 = cv2.cvtColor(sementes1, cv2.COLOR_BGR2GRAY)
            
            plt.imshow(gaussino,'gray')#certa para visualizar imagem
            plt.title('teste')
            plt.show()
          
            imgROI = imgROI1[550:3400,550:3400]
            gaussino = gaussino[550:3400,550:3400]
            plt.imshow(gaussino,'gray')
            plt.title("filtro")
            plt.show()
            plt.imshow(imgROI,'gray')
            plt.title("imagem")
            plt.show()
            # cv2.imwrite("E:/Dissertacao/Imagens/FrequenciaIdeal_Imagem_Original/"+str(folder)+"/"+str(file) + ".bmp", gaussino)
            d1 = np.mean((imgROI - gaussino) ** 2, dtype=np.float64)
            d = 10 * np.log10((255 ** 2) / d1)
            d2 = ssim4(imgROI.astype(np.uint8), gaussino.astype(np.uint8),multichannel=True)
            
            with open('E:/Dissertacao/Resultados_Metricas/Filtro1/FrequenciaIdeal/'+str(folder)+'/psnr/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d))
            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro1/FrequenciaIdeal/'+str(folder)+'/mse/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d1))
            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro1/FrequenciaIdeal/'+str(folder)+'/ssim/'+str(folder)+str(file)+'.txt','w') as arq2:
                arq2.write(str(d2))
            del(arq2)           
            
            print('imagem com ruido sal e pimenta com add 0.05')
            salEpimenta1 = cv2.cvtColor(salEpimenta1, cv2.COLOR_BGR2GRAY)  
            gaussino_ruidoAlto=filtroFrequenciaAltaIdealLivro(salEpimenta1)  
            baixarfeq_ruido=filtroFrequenciaBaixaIdealLivro(salEpimenta1)
            frequencia_ruido=filtroFrequenciaBandaIdealLivro(gaussino_ruidoAlto,baixarfeq_ruido)
            frequencia_ruido=cv2.bitwise_and(frequencia_ruido,frequencia_ruido,mask=img1)
            
            frequencia_ruido = frequencia_ruido[550:3400,550:3400]
                 
            salEpimenta1=cv2.bitwise_and(salEpimenta1,salEpimenta1,mask=img1)
           
            salEpimenta1 = salEpimenta1[550:3400,550:3400]
            plt.imshow(salEpimenta1,'gray')
            plt.title(' ruido')
            plt.show()
            
            plt.imshow(frequencia_ruido,'gray')
            plt.title('frequencia ruido')
            plt.show()
            
            plt.imshow(imgROI,'gray')
            plt.title('frequencia sem ruido')
            plt.show()
            
            print('freq na imagem ruidosa')  
            
            # cv2.imwrite("E:/Dissertacao/Imagens/FrequenciaIdeal_Imagem_Ruido/"+str(folder)+"/"+str(file) + ".bmp", frequencia_ruido)
                               
            d1 = np.mean((imgROI - frequencia_ruido) ** 2, dtype=np.float64)
            d = 10 * np.log10((255 ** 2) / d1)
            d2 = ssim4(imgROI.astype(np.uint8), frequencia_ruido.astype(np.uint8),multichannel=True)
            
            with open('E:/Dissertacao/Resultados_Metricas/Filtro1/FrequenciaIdeal_ruido/'+str(folder)+'/psnr/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d))
            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro1/FrequenciaIdeal_ruido/'+str(folder)+'/mse/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d1))
            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro1/FrequenciaIdeal_ruido/'+str(folder)+'/ssim/'+str(folder)+str(file)+'.txt','w') as arq2:
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
