import numpy as np
import cv2
import random as r
import os
from skimage.measure import compare_ssim as ssim
def filtro_gaussiano():
    raiz ="/Diretorio/.../" 
    amostra = 0
    for pasta in os.listdir(raiz):
        arquivos = os.listdir(raiz + pasta)
        for arquivo in arquivos:
            sementes= cv2.imread(raiz + pasta + "/" + arquivo)  
            sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
            imagem_mascara= np.zeros(sementes.shape[:2],dtype="uint8")
            (cX,cY)=(sementes.shape[0] // 2,sementes.shape[1] // 2)
            cv2.circle(img1,(2000,2000),1395,255,-1)
            imagem_roi=cv2.bitwise_and(sementes,sementes,mask=imagem_mascara)
            imagem_ruido_gaussiano = ruido_gaussiano(imagem_roi,0.3,0)
            imagem_ruido_impulsivo = ruido_impulsivo(imagem_roi,0.05)
            imagem_filtro_gaussiano=cv2.GaussianBlur(imagem_ruido_gaussiano,(5,5),0)
            imagem_filtro_impulsivo=cv2.GaussianBlur(imagem_ruido_impulsivo,(5,5),0)
            mse_gaussiano = mse(imagem_roi,imagem_filtro_gaussiano)
            psnr_gaussiano = psnr(imagem_roi,imagem_filtro_gaussiano)
            ssim_gaussiano = ssim (imagem_roi,imagem_filtro_gaussiano,multichannel=False)           
			mse_impulsivo = mse(imagem_roi,imagem_filtro_impulsivo )
            psnr_impulsivo = psnr(imagem_roi,imagem_filtro_impulsivo )
            ssim_impulsivo = ssim (imagem_roi,imagem_filtro_impulsivo ,multichannel=False)           
            amostra = amostra+1      
           
    return amostra