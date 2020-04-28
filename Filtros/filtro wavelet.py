# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 00:50:21 2020

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
import pywt
import wtTools as wtt
from skimage.restoration import denoise_nl_means
from skimage.measure import compare_psnr as psnr4

from skimage.measure import compare_mse as mse4

from skimage.measure import compare_ssim as ssim4
import timeit
import time 
import scipy.fftpack as fftpack

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



    



    

def ler():
    #raiz = 'C:/Users/andre/leitura_teste/'

    
    raiz ='E:/Dissertacao/Imagens/imagem original/'
    #raiz ='C:/Users/andre/OneDrive/Documentos/Base_de_dados_com_100_cada_semente/imagem_ROI1/'
    
    lista = []
    amostra = 1
    id = []
    cont = 0
    soma = 1
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
            
            wav = 'db3'
            NLEV  = 4
            filter_bank = pywt.Wavelet(wav)
            coeffs = pywt.wavedec2(imgROI, filter_bank, mode='per', level=NLEV)
            arr = wtt.coeffs_to_array(coeffs)
            
            # difference between wtview and imshow?
            wtt.wtView(coeffs, 'Original image, scaled DWT coeffs')
            plt.imshow(np.abs(arr/arr.max()))
            plt.set_cmap('jet')
            plt.title('Unscaled DWT coefficients')
            plt.show()
            
            #%% Inverse DWT
            decoded = pywt.waverec2(coeffs, filter_bank, mode='per')
            plt.imshow(np.uint8(decoded))
            plt.title('Inverse DWT') ,  plt.set_cmap('gray'), plt.show()
            
            
          
            
            
            
            
            amostra = amostra + 1
            cont = cont + 1
            soma =soma +1 
    #arq2.close()
    #arq3.close()
    #os.close(comp1)

    return lista


t = ler()

print(t)

cv2.waitKey(0)
cv2.destroyAllWindows()
