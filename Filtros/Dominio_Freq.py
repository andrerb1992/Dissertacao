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
import sys
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



    
def filtroGaussianoBandaLivro(im):
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    fig, (axes1, axes2) = pylab.subplots(1, 2, figsize=(10,5))
    pylab.gray() # show the result in grayscale
    im = np.mean(im,axis=2)
    freq = fp.fft2(im)
    freq_gaussian = ndimage.fourier_gaussian(freq, sigma=15)
    im1 = fp.ifft2(freq_gaussian)
    axes1.imshow(im), axes1.axis('off'), axes2.imshow(im1.real) # the imaginary part is an artifact
    axes2.axis('off')
    pylab.show()
    
    plt.imshow(im1.astype(np.uint8), cmap='gray')
    plt.title('gaussiano baixa1')
    plt.show()
    
    freq_gaussiana = ndimage.fourier_gaussian(freq, sigma=0.05)
    freq_gaussian1 = 1 - freq_gaussiana
    im_alta = fp.ifft2(freq_gaussian1)
    axes1.imshow(im), axes1.axis('off'), axes2.imshow(im_alta.real) # the imaginary part is an artifact
    axes2.axis('off')
    pylab.show()
    
    plt.imshow(im_alta.astype(np.uint8), cmap='gray')
    plt.title('gaussiano alta')
    plt.show()
            
    passsa_banda = im_alta.astype(np.uint8) * im1.astype(np.uint8)
    plt.imshow(passsa_banda.astype(np.uint8), cmap='gray')
    plt.title('gaussiano banda')
    plt.show()   
            
    return passsa_banda.astype(np.uint8)


def filtroGaussianoBaixaLivro(im):
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    fig, (axes1, axes2) = pylab.subplots(1, 2, figsize=(10,5))
    pylab.gray() # show the result in grayscale
    im = np.mean(im,axis=2)
    freq = fp.fft2(im)
    freq_gaussian = ndimage.fourier_gaussian(freq, sigma=10)
    im1 = fp.ifft2(freq_gaussian)
    axes1.imshow(im), axes1.axis('off'), axes2.imshow(im1.real) # the imaginary part is an artifact
    axes2.axis('off')
    pylab.show()
    plt.imshow(im1.astype(np.uint8), cmap='gray')
    plt.title('gaussiano baixa')
    plt.show()            
    return im1.astype(np.uint8)

def filtroGaussianoAltaLivro(im):
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    fig, (axes1, axes2) = pylab.subplots(1, 2, figsize=(10,5))
    pylab.gray() # show the result in grayscale
    im = np.mean(im,axis=2)
    freq = fp.fft2(im)   
    freq_gaussiana = ndimage.fourier_gaussian(freq, sigma=0.05)
    freq_gaussian1 = 1 - freq_gaussiana
    im_alta = fp.ifft2(freq_gaussian1)
    axes1.imshow(im), axes1.axis('off'), axes2.imshow(im_alta.real) # the imaginary part is an artifact
    axes2.axis('off')
    pylab.show()
    
    plt.imshow(im_alta.astype(np.uint8), cmap='gray')
    plt.title('gaussiano alta')
    plt.show()            
    return im_alta.astype(np.uint8)


    

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
            #cv2.imwrite("C:/Users/andre.brito.INSTRUMENTACAO/Desktop/Base_de_dados_com_100_cada_semente/imagem_ROI/"+str(folder)+"/"+str(file) + ".bmp", imgROI)
            imgROI1 = cv2.cvtColor(imgROI, cv2.COLOR_BGR2GRAY) 
            filtropassa_faixa_gaussiano =  filtroGaussianoBandaLivro(imgROI.astype(np.uint8))
            cv2.imwrite("E:/Dissertacao/Imagens/Frequencia_Imagem_Original/"+str(folder)+"/"+str(file) + ".bmp", filtropassa_faixa_gaussiano)
            print(imgROI.shape)
            print(filtropassa_faixa_gaussiano.shape)
            d = psnr4(imgROI1.astype(np.uint8), filtropassa_faixa_gaussiano.astype(np.uint8))
            d1 = mse4(imgROI1.astype(np.uint8), filtropassa_faixa_gaussiano.astype(np.uint8))
            d2 = ssim4(imgROI1.astype(np.uint8), filtropassa_faixa_gaussiano.astype(np.uint8),multichannel=True)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Frequencia/'+str(folder)+'/psnr/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d))
            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Frequencia/'+str(folder)+'/mse/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d1))
            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Frequencia/'+str(folder)+'/ssim/'+str(folder)+str(file)+'.txt','w') as arq2:
                arq2.write(str(d2))
            del(arq2)  
          
            print('imagem com ruido sal e pimenta com add 0.05')
            salEpimenta1 =sp_noise(imgROI.astype(np.uint8), 0.005)
 
                      

            filtropassa_faixa_gaussiano_ruido = filtroGaussianoBandaLivro(salEpimenta1.astype(np.uint8))
            cv2.imwrite("E:/Dissertacao/Imagens/Frequencia_Imagem_Ruido/"+str(folder)+"/"+str(file) + ".bmp", filtropassa_faixa_gaussiano_ruido)
            d = psnr4(imgROI1.astype(np.uint8), filtropassa_faixa_gaussiano_ruido.astype(np.uint8))
            d1 = mse4(imgROI1.astype(np.uint8), filtropassa_faixa_gaussiano_ruido.astype(np.uint8))
            d2 = ssim4(imgROI1.astype(np.uint8), filtropassa_faixa_gaussiano_ruido.astype(np.uint8),multichannel=True)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Frequencia_ruido/'+str(folder)+'/psnr/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d))
            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Frequencia_ruido/'+str(folder)+'/mse/'+str(file)+'.txt','w') as arq2:
                arq2.write(str(d1))
            del(arq2)
            with open('E:/Dissertacao/Resultados_Metricas/Filtro/Frequencia_ruido/'+str(folder)+'/ssim/'+str(folder)+str(file)+'.txt','w') as arq2:
                arq2.write(str(d2))
            del(arq2)  
           
            
          
            
            
            
            
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
