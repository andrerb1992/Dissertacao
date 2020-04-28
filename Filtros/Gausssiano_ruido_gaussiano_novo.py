# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 13:13:15 2020

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

def noisy(image):    
    row,col,ch= image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy



def processo_ruido():

    raiz ='E:/Dissertacao/RuidoGaussiano/Imagens/Imagens_ruidosa/'    
    lista = []
    amostra = 1
    id = []
    soma = 1
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
            print('processo_ruido na imagem original')             
            gaussino=cv2.GaussianBlur(sementes,(5,5),0)
            gaussino=cv2.bitwise_and(gaussino,gaussino,mask=img1)
            gaussino = cv2.cvtColor(gaussino, cv2.COLOR_BGR2GRAY)            
            sementes1=cv2.bitwise_and(sementes,sementes,mask=img1)           
            imgROI1 = cv2.cvtColor(sementes1, cv2.COLOR_BGR2GRAY)     
            imgROI = imgROI1[550:3400,550:3400]
            gaussino = gaussino[550:3400,550:3400]
            plt.imshow(gaussino,'gray')
            plt.title("a")
            plt.show()
            # cv2.imwrite("E:/Dissertacao/RuidoGaussiano/Imagens/Gaussiano_ruido/"+str(folder)+"/"+str(file) + ".bmp", gaussino)
            d = psnr4(imgROI.astype(np.uint8), gaussino.astype(np.uint8))
            d1 = mse4(imgROI.astype(np.uint8), gaussino.astype(np.uint8))
            d2 = ssim4(imgROI.astype(np.uint8), gaussino.astype(np.uint8),multichannel=True)
            
            d1 = np.mean((imgROI - gaussino) ** 2, dtype=np.float64)
            d = 10 * np.log10((255 ** 2) / d1)
            d2 = ssim4(imgROI.astype(np.uint8), gaussino.astype(np.uint8),multichannel=True)
            
            with open('E:/Dissertacao/RuidoGaussiano/Metrica1/Gaussiano_ruido/'+str(folder)+'/psnr/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d))
            del(arq2)
            with open('E:/Dissertacao/RuidoGaussiano/Metrica1/Gaussiano_ruido/'+str(folder)+'/mse/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d1))
            del(arq2)
            with open('E:/Dissertacao/RuidoGaussiano/Metrica1/Gaussiano_ruido/'+str(folder)+'/ssim/'+str(folder)+str(file)+'.txt','w') as arq2:
                arq2.write(str(d2))
            del(arq2) 
            
            print('imagem com ruido sal e pimenta')
           
            amostra = amostra + 1                
            soma =soma +1 
    return lista


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
            
            gaussino=cv2.GaussianBlur(sementes,(5,5),0)
            gaussino=cv2.bitwise_and(gaussino,gaussino,mask=img1)
            gaussino = cv2.cvtColor(gaussino, cv2.COLOR_BGR2GRAY)
            
            sementes1=cv2.bitwise_and(sementes,sementes,mask=img1)           
            imgROI1 = cv2.cvtColor(sementes1, cv2.COLOR_BGR2GRAY)
            
            #dst = cv2.cvtColor(gaussino, cv2.COLOR_BGR2GRAY)
            imgROI = imgROI1[550:3400,550:3400]
            gaussino = gaussino[550:3400,550:3400]
            plt.imshow(gaussino,'gray')
            plt.title("a")
            plt.show()
            cv2.imwrite("E:/Dissertacao/RuidoGaussiano/Imagens/Gaussiano/"+str(folder)+"/"+str(file) + ".bmp", gaussino)
            d = psnr4(imgROI.astype(np.uint8), gaussino.astype(np.uint8))
            d1 = mse4(imgROI.astype(np.uint8), gaussino.astype(np.uint8))
            d2 = ssim4(imgROI.astype(np.uint8), gaussino.astype(np.uint8),multichannel=True)
            
            with open('E:/Dissertacao/RuidoGaussiano/Metrica/Gaussiano/'+str(folder)+'/psnr/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d))
            del(arq2)
            with open('E:/Dissertacao/RuidoGaussiano/Metrica/Gaussiano/'+str(folder)+'/mse/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d1))
            del(arq2)
            with open('E:/Dissertacao/RuidoGaussiano/Metrica/Gaussiano/'+str(folder)+'/ssim/'+str(folder)+str(file)+'.txt','w') as arq2:
                arq2.write(str(d2))
            del(arq2) 
            
            print('imagem com ruido sal e pimenta')
#            salEpimenta1 =noisy(sementes) 
#            cv2.imwrite("E:/Dissertacao/RuidoGaussiano/Imagens/Imagens_ruidosa/"+str(folder)+"/"+str(file) + ".bmp", salEpimenta1)
#            
                    
               
           
            

            amostra = amostra + 1
                
            soma =soma +1 
    #arq2.close()
    #arq3.close()
    #os.close(comp1)

    return lista


#t = ler()
#
#print(t)

process = processo_ruido()

print(process)


cv2.waitKey(0)
cv2.destroyAllWindows()
