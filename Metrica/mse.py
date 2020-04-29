import numpy as np
import cv2
import random as r
def mse(imagem,imagem_filtrada):
    mse_resultado = 0
    valor = 0
    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            aux = ((imagem[i][j] - imagem_filtrada[i][j]) ** 2)
            valor = valor + aux
    mse_resultado = valor/(imagem.shape[0] * imagem.shape[1])
    print(mse_resultado)
    return mse_resultado