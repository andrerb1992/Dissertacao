# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 04:02:53 2020

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
from skimage.restoration import denoise_nl_means, estimate_sigma
import numpy.fft as fp
import scipy.fftpack as fp
import scipy.signal
import scipy.ndimage
import sys
from skimage.restoration import denoise_nl_means
from skimage.measure import compare_psnr as psnr4

from skimage.measure import compare_mse as mse4

from skimage.measure import compare_ssim as ssim4

from skimage import data, io, filters

def transformar5 (img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j]==0:
                img[i][j]=0
            elif img[i][j]==1:
                img[i][j]=255
    return img 



def ler():
    #raiz = 'C:/Users/andre/leitura_teste/'

    
    raiz ='E:/Dissertacao/Imagens/imagem original/'
    #raiz ='C:/Users/andre/OneDrive/Documentos/Base_de_dados_com_100_cada_semente/imagem_ROI1/'
    
    lista = []
 
    id = []
   
    for folder in os.listdir(raiz):
        # print(folder)
        files = os.listdir(raiz + folder)
        print(folder)
        # print (files)

        

        id.append(folder + " ")
        
        for file in files:
            # if '.bmp' in file:
            lista.append(raiz + folder + '/' + file)
            sementes = cv2.imread(raiz + folder + '/' + file)
            # sementes = cv2.cvtColor(sementes,cv2.COLOR_BGR2GRAY)

            #plt.imshow(sementes, 'gray')
            #plt.title(folder)
            #plt.show()
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
            #cv2.imwrite("C:/Users/andre.brito.INSTRUMENTACAO/Desktop/Base_de_dados_com_100_cada_semente/imagem_ROI/"+str(folder)+"/"+str(file) + ".bmp", imgROI)
            sementes=cv2.GaussianBlur(imgROI,(3,3),0)  
            imgROI1 = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
            
            threshold = filters.threshold_otsu(imgROI1)
            thresh1 = imgROI1>= threshold
            plt.imshow(thresh1, 'gray')
            plt.title(folder+"2 otsu")
            plt.show()               
            mask = np.ones(imgROI1.shape,np.uint8) #crio uma mascara so de 1
            a = mask * thresh1 # tranforma para binario a imagem                
            thresh12 = transformar5(a)
            imagem_resultado= imgROI1.astype(np.uint8) * thresh1.astype(np.uint8)
            plt.imshow(thresh12, 'gray')
            plt.title("imagem_resultado")
            plt.show() 
           
            cv2.imwrite("E:/Dissertacao/Imagens/imagem original_defeituosa/"+str(folder)+"/"+str(file) + ".bmp", thresh12)
            
            
          
            
            
            
            
    #arq2.close()
    #arq3.close()
    #os.close(comp1)

    return lista


t = ler()

print(t)

cv2.waitKey(0)
cv2.destroyAllWindows()

