import numpy as np
import cv2
import random as r
def psnr (imagem,imagem_filtrada):
    psnr_resultado = 0
    valor = 0
	mse_resultado = 0
    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            aux = ((imagem[i][j] - imagem_filtrada[i][j]) ** 2)
            valor = valor + aux
    mse_resultado = valor/(imagem.shape[0] * imagem.shape[1])
	psnr_resultado = 10 * np.log10((255 ** 2) / mse_resultado)
    print(psnr_resultado)
    return psnr_resultado