import numpy as np
import cv2
import random as r
import os
import numpy.fft as fp
#from scipy.fftpack import fft2, ifft2, fftfreq, fftshift
from skimage.measure import compare_ssim as ssim

def filtroFrequenciaAltaIdeal(imagem, valorA, valorB):
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    freq = fp.fft2(imagem)
    (linha, coluna) = freq.shape
    half_linha, half_coluna = int(linha/2), int(coluna/2)
    freq1 = np.copy(freq)
    freq2 = fp.fftshift(freq1)
    freq2[half_linha-valorA:half_linha+valorB,half_coluna-valorA:half_coluna+valorB] = 0 
    filtro_passa_alta = 1 - freq2
    filtro_passa_alta= fp.ifft2(fftpack.ifftshift(filtro_passa_alta).real)
    return filtro_passa_baixa

def filtroFrequenciaBaixaIdeal(imagem, valorA, valorB):
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    freq = fp.fft2(imagem)
    (linha, coluna) = freq.shape
    half_linha, half_coluna = int(linha/2), int(coluna/2)
    freq1 = np.copy(freq)
    freq2 = fp.fftshift(freq1)
    freq2[half_linha-valorA:half_linha+valorB,half_coluna-valorA:half_coluna+valorB] = 0 
    filtro_passa_alta = fp.ifft2(fftpack.ifftshift(freq2).real,0,)
    return filtro_passa_baixa
    
def filtroFrequenciaBandaIdeal(imagem1,imagem2):
    imagem1_alta=filtroFrequenciaAltaIdeal(imagem1,5,6)
    imagem1_baixa=filtroFrequenciaBaixaIdeal(imagem2,24,25)
    resultado = imagem1_alta * imagem2_baixa
    return resultado
    
def filtro_frequencia():
    raiz ="/Diretorio/.../" 
    amostra = 0
    for pasta in os.listdir(raiz):
        arquivos = os.listdir(raiz + pasta)
        for arquivo in arquivos:
            sementes= cv2.imread(raiz + pasta + "/" + arquivo)  
            #sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
            imagem_mascara= np.zeros(sementes.shape[:2],dtype="uint8")
            (cX,cY)=(sementes.shape[0] // 2,sementes.shape[1] // 2)
            cv2.circle(img1,(2000,2000),1395,255,-1)
            imagem_roi=cv2.bitwise_and(sementes,sementes,mask=imagem_mascara)
            imagem_ruido_gaussiano = ruido_gaussiano3d(imagem_roi,0.3,0)
            imagem_ruido_impulsivo = ruido_impulsivo(imagem_roi,0.05)
            imagem_roi = cv2.cvtColor(imagem_roi, cv2.COLOR_BGR2GRAY)
            imagem_ruido_gaussiano = cv2.cvtColor(imagem_ruido_gaussiano, cv2.COLOR_BGR2GRAY)
            imagem_ruido_impulsivo= cv2.cvtColor(imagem_ruido_impulsivo, cv2.COLOR_BGR2GRAY)
            imagem_filtro_gaussiano=filtroFrequenciaBandaIdeal(imagem_ruido_gaussiano)
            imagem_filtro_impulsivo=filtroFrequenciaBandaIdeal(imagem_ruido_impulsivo)
            mse_gaussiano = mse(imagem_roi,imagem_filtro_gaussiano)
            psnr_gaussiano = psnr(imagem_roi,imagem_filtro_gaussiano)
            ssim_gaussiano = ssim (imagem_roi,imagem_filtro_gaussiano,multichannel=False)
            mse_impulsivo = mse(imagem_roi,imagem_filtro_impulsivo )
            psnr_impulsivo = psnr(imagem_roi,imagem_filtro_impulsivo )
            ssim_impulsivo = ssim (imagem_roi,imagem_filtro_impulsivo ,multichannel=False)
            
           
    return amostra
    