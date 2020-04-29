import numpy as np
import random as r
def ruido_gaussiano(imagem, sigma, mean):    
    linha= imagem.shape[0]
    coluna= imagem.shape[1]
    gaussiano = np.random.normal(mean,sigma,(linha,coluna))
    gaussiano = gaussiano.reshape(linha,coluna)
    imagem_ruidosa = imagem + gaussiano
    return imagem_ruidosa
    
def ruido_gaussiano3d(imagem, sigma, mean):    
    linha= imagem.shape[0]
    coluna= imagem.shape[1]
    profundidade= imagem.shape[2]
    gaussiano = np.random.normal(mean,sigma,(linha,coluna,profundidade))
    gaussiano = gaussiano.reshape(linha,coluna,profundidade)
    imagem_ruidosa = imagem + gaussiano
    return imagem_ruidosa