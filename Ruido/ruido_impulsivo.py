import numpy as np
import random as r
def ruido_impulsivo(imagem, prob):
    imagem_ruidosa = np.zeros(imagem.shape, np.uint8)
    aux = 1 - prob
    for i in range(image.shape[0]):
        for j in range(imagem.shape[1]):
            gerar_numero = r.random()
            if gerar_numero < prob:
                imagem_ruidosa[i][j] = 0
            elif gerar_numero > aux:
                imagem_ruidosa[i][j] = 255
            else:
                imagem\_ruidosa[i][j] = imagem[i][j]
    return imagem_ruidosa