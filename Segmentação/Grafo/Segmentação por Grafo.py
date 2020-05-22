import os
import math
import numpy as np
import networkx as nx
from scipy.spatial import distance
# a funcao ford_fulkerson nao existe mais na versao 2.2
#from networkx.algorithms.flow import ford_fulkerson
from networkx.algorithms.flow import maximum_flow, edmonds_karp
from skimage import io
from skimage.transform import resize
#from skimage.util import random_noise
#from scipy.misc import imsave

def janela(p, imagem, comprimento = 3, altura = 3):
    
    # p[0] = l = linha; p[1] = c = coluna
    l, c = p[0], p[1]
    dim = np.shape(imagem)
    nlin, ncol = dim[0], dim[1]
#   print("ncol = " + str(ncol) + ", nlin = " + str(nlin))
    offset_alt = altura // 2
    offset_comp = comprimento // 2
    li = lf = ci = cf = 0
    
    # linha inferior (li)
    if ( l - offset_alt ) < 0:
        li = 0
    else:
        li = l - offset_alt
        
    # linha superior (ls)
    if ( l + offset_alt ) > nlin:
        lf = l
    else:
        lf = l + offset_alt + 1
        
    # coluna inferior (ci)
    if ( c - offset_comp ) < 0:
        ci = 0
    else:
        ci = c - offset_comp
        
    # coluna superior (cs)
    if ( c + offset_comp ) > ncol:
        cf = c
    else:
        cf = c + offset_comp + 1
        
    jan = imagem[ li:lf, ci:cf ]
    
    return jan    

def desvioPadrao(p, imagem):
    
    j = janela(p, imagem)
    desvio = np.std(j)
    return desvio
    
def getVariancia(imagem):
    
    return np.var(imagem)
    
def getIntensidade(p, imagem):
    """Retorna a intensidade (tom de cinza) de um ponto "p" na imagem "imagem""""
    l, c = p[0], p[1]
    return imagem[l][c]
    
def distancia(p, q, imagem):
    
    intp, intq = getIntensidade(p, imagem), getIntensidade(q, imagem)
    desvp, desvq = desvioPadrao(p, imagem), desvioPadrao(q, imagem)
    
    ptemp, qtemp = (intp, desvp), (intq, desvq)

    distancia = distance.euclidean(ptemp, qtemp)
    
    return distancia
    
def beta(imagem, p, q, variancia = 1):
   
    beta, d = 0, distancia(p, q, imagem)
    if d > 0:
        intp, intq = getIntensidade(p, imagem), getIntensidade(q, imagem)
        numerador = -1 * ((intp - intq) ** 2)
        denominador = 2 * variancia
        termo = 1 / d
        beta =  math.exp( numerador / denominador ) * termo       
    return beta   
    
def mediaCinza(p, imagem, comprimento = 3, largura = 3):
    
    j = janela(p, imagem, comprimento, largura)
    media = np.sum(j) / (comprimento * largura)
    return media


def criaDigrafo(imagem, media_s, media_t, ps, pt, s="s", t="t", fn = beta, var = -1):
   
    dim = np.shape(imagem)
    linhas, colunas = dim[0], dim[1]
    G = nx.DiGraph()
    
    variancia = var
    if variancia < 0:
        variancia = getVariancia(imagem)

    for i in range(0, linhas):
        for j in range(0, colunas):
            if i>0:
                # Conecta para cima
                G.add_edge(str(i)+"_"+str(j), str(i-1)+"_"+str(j), capacity = fn( imagem, (i,j), (i-1, j), variancia ) )
            if i<(linhas-1):
                # Conecta para baixo
                G.add_edge(str(i)+"_"+str(j), str(i+1)+"_"+str(j), capacity = fn( imagem, (i,j), (i+1, j), variancia ) )
            if j>0:
                # Conecta para esquerda
                G.add_edge(str(i)+"_"+str(j), str(i)+"_"+str(j-1), capacity = fn( imagem, (i,j), (i, j-1), variancia ) )
            if j<(colunas-1):
                # Conecta para direita
                G.add_edge(str(i)+"_"+str(j), str(i)+"_"+str(j+1), capacity = fn( imagem, (i,j), (i, j+1), variancia ) )
                
  
    ponto_s = ps
    ponto_t = pt
    var_s = getVariancia(janela(ponto_s, imagem, 5, 5))
    var_t = getVariancia(janela(ponto_t, imagem, 5, 5))
           
    # inserir o vertice s e t
    for i in range(0, linhas):
        for j in range(0, colunas):
            G.add_edge(s, str(i)+"_"+str(j), capacity = verossimilhanca(im[i][j], media_s, var_s) )
            G.add_edge(str(i)+"_"+str(j), t, capacity = verossimilhanca(im[i][j], media_t, var_t) )
            
    return G
    
def ff(G, s = "s", t = "t"):
    
    _, flow_dict = maximum_flow( G, s, t, "capacity", flow_func=edmonds_karp) 
    node_s = flow_dict[s]
    nos_dict = {}
    
    # o BFS so no primeiro nivel
    for v,k in node_s.items():
        if k > 1e-10:
            nos_dict[v] = k
    
    return flow_dict, nos_dict
    
    
def segmentar(grupoA, comprimento , altura , s = "s"):
    resultado = np.ones((altura, comprimento), dtype=np.int8)

    if not grupoA:
        return resultado
        
#    vertices = grupoA[s]
#    
#    if not vertices:
#        return resultado
#
#    for i in range(0, len(vertices)):
#        rotulo = vertices[i].split("_")
#        l = int(rotulo[0])
#        c = int(rotulo[1])
#        resultado[l][c] = 0
    
    for v,_ in grupoA.items():
        rotulo = v.split("_")
        l = int(rotulo[0])
        c = int(rotulo[1])
        resultado[l][c] = 0
        
    return resultado
    

def modulo(imagem, p, q, x = 0):
    """Retorna 1 - abs(p - q)"""
    lp, cp = p
    lq, cq = q
    m = 255 - abs(imagem[lp][cp] - imagem[lq][cq])
    return m


def condgradient(delta, kappa, spacing, option = 1):
    if option == 1:
        return np.exp(-(delta/kappa)**2.)/float(spacing)
    elif option == 2:
        return 1./(1.+(delta/kappa)**2.)/float(spacing)
        
        
def conduction(imagem, p, q, x = 0):
    lp, cp = p
    lq, cq = q
    deltai = abs(imagem[lp][cp] - imagem[lq][cq])
    c = condgradient(deltai, 80, 1.0, 1)
    return c * 255
    
def verossimilhanca(valor, media, variancia):
    vero = (1 / math.sqrt(2 * math.pi * variancia)) * math.exp( (-1/(2*variancia)) * math.pow( (valor - media), 2)) 
    return vero
    
    
def coeficenteKappa(verdade_terrestre, imagem):
    # A - verdade terrestre => linhas
    # B - imagem => colunas
    # Matriz de confusao
    # [0][0] - objeto/objeto => objeto nas duas imagens (objeto verdadeiro)
    # [0][1] - objeto/fundo => objeto na verdade e fundo na imagem (fundo falso)
    # [1][0] - fundo/objeto => fundo na verdade e objeto na imagem (objeto falso)
    # [1][1] - fundo/fundo => fundo nas duas imagens (fundo verdadeiro)
    matriz_confusao = np.array([[0,0], [0,0]])
    dim = np.shape(verdade_terrestre)
    linhas, colunas = dim[0], dim[1]
    for l in range(0,linhas):
        for c in range(0,colunas):
            voriginal = verdade_terrestre[l][c]
            vimagem = imagem[l][c]
            if voriginal == 0: # objeto
                if vimagem == 0:
                    matriz_confusao[0][0] = matriz_confusao[0][0] + 1
                else:
                    matriz_confusao[0][1] = matriz_confusao[0][1] + 1
            elif vimagem == 1: # considero que voriginal eh 1
                matriz_confusao[1][1] = matriz_confusao[1][1] + 1
            else:
                matriz_confusao[1][0] = matriz_confusao[1][0] + 1
                
    somacolunas = matriz_confusao.sum(axis = 0)
    somalinhas = matriz_confusao.sum(axis = 1)

    N = linhas * colunas    
    termo = (somalinhas[0] * somacolunas[0]) + (somalinhas[1] * somacolunas[1])
    total_acertos = matriz_confusao[0][0] + matriz_confusao[1][1]
    
    kappa = ((N * total_acertos) - termo)/( (N ** 2) - termo)
    
    return kappa

def grafo():
    raiz =".../Imagens/"
    for pasta in os.listdir(raiz):
        arquivos = os.listdir(raiz + pasta)
        soma = 1
        for arquivo in arquivos:            
            sementes = cv2.imread(raiz + pasta + "/" + arquivo)
            if pasta == "Girassol 1_Rec":                
                sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                ponto_s = (1310,560)
                ponto_t = (740,1540)
                imagem = cv2.equalizeHist(sementes)
                media_s = seg.mediaCinza(ponto_s, imagem, 5, 5)
                media_t = seg.mediaCinza(ponto_t, imagem, 5, 5)
                G = seg.criaDigrafo(imagem, media_s, media_t, ponto_s, ponto_t,"s", "t", seg.modulo )
                flow_dict, nos_dict = seg.ff(G, "s")              
                imagem_resultado = seg.segmentar(nos_dict, imagem.shape[0], imagem.shape[1])
                imagem_normalizada = ((imagem_resultado  - np.min(imagem_resultado )) / (np.max(imagem_resultado) - np.min(imagem_resultado))) * 256
                cv2.imwrite("C.../Segmentacao/Grafo/Imagens/"+str(pasta)+"/"+str(arquivo) + ".bmp", imagem_normalizada)
                imagem_resultante = imagem_normalizada * imagem
                resultado_ssim= ssim(imagem,imagem_resultante,multichannel=True)
                with open (".../Segmentacao/Grafo/Resultados_Metricas/"+str(pasta)+"/ssim/"+str(arquivo)+".txt","w") as arq2:36arq2.write(str(resultado_ssim))
           
            if pasta == "Girassol 2_Rec":                
                sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                ponto_s = (720,770)
                ponto_t = (950,590) 
                imagem = cv2.equalizeHist(sementes)
                media_s = seg.mediaCinza(ponto_s, imagem, 5, 5)
                media_t = seg.mediaCinza(ponto_t, imagem, 5, 5)
                G = seg.criaDigrafo(imagem, media_s, media_t, ponto_s, ponto_t,"s", "t", seg.modulo )
                flow_dict, nos_dict = seg.ff(G, "s")              
                imagem_resultado = seg.segmentar(nos_dict, imagem.shape[0], imagem.shape[1] )
                imagem_normalizada = ((imagem_resultado  - np.min(imagem_resultado )) / (np.max(imagem_resultado) - np.min(imagem_resultado))) * 256
                cv2.imwrite("C.../Segmentacao/Grafo/Imagens/"+str(pasta)+"/"+str(arquivo) + ".bmp", imagem_normalizada)
                imagem_resultante = imagem_normalizada * imagem
                resultado_ssim = ssim(imagem,imagem_resultante,multichannel=True)
                with open (".../Segmentacao/Grafo/Resultados_Metricas/"+str(pasta)+"/ssim/"+str(arquivo)+".txt","w") as arq2:36arq2.write(str(resultado_ssim))                                            
            
            if pasta == "Girassol 3_Rec":
                sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                ponto_s = (450,1780)
                ponto_t = (750,950) 
                imagem = cv2.equalizeHist(sementes)
                media_s = seg.mediaCinza(ponto_s, imagem, 5, 5)
                media_t = seg.mediaCinza(ponto_t, imagem, 5, 5)
                G = seg.criaDigrafo(imagem, media_s, media_t, ponto_s, ponto_t,"s", "t", seg.modulo )
                flow_dict, nos_dict = seg.ff(G, "s")              
                imagem_resultado = seg.segmentar(nos_dict, imagem.shape[0], imagem.shape[1] )
                imagem_normalizada = ((imagem_resultado  - np.min(imagem_resultado )) / (np.max(imagem_resultado) - np.min(imagem_resultado))) * 256
                cv2.imwrite("C.../Segmentacao/Grafo/Imagens/"+str(pasta)+"/"+str(arquivo) + ".bmp", imagem_normalizada)
                imagem_resultante = imagem_normalizada * imagem
                resultado_ssim= ssim(imagem,imagem_resultante,multichannel=True)
                with open (".../Segmentacao/Grafo/Resultados_Metricas/"+str(pasta)+"/ssim/"+str(arquivo)+".txt","w") as arq2:36arq2.write(str(resultado_ssim))
    
            if pasta == "Girassol 4_Rec":
                sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                ponto_s = (830,1070)
                ponto_t = (1260,90) 
                imagem = cv2.equalizeHist(sementes)
                media_s = seg.mediaCinza(ponto_s, imagem, 5, 5)
                media_t = seg.mediaCinza(ponto_t, imagem, 5, 5)
                G = seg.criaDigrafo(imagem, media_s, media_t, ponto_s, ponto_t,"s", "t", seg.modulo )
                flow_dict, nos_dict = seg.ff(G, "s")              
                imagem_resultado = seg.segmentar(nos_dict, imagem.shape[0], imagem.shape[1] )
                imagem_normalizada = ((imagem_resultado  - np.min(imagem_resultado )) / (np.max(imagem_resultado) - np.min(imagem_resultado))) * 256
                cv2.imwrite("C.../Segmentacao/Grafo/Imagens/"+str(pasta)+"/"+str(arquivo) + ".bmp", imagem_normalizada)
                imagem_resultante = imagem_normalizada * imagem
                resultado_ssim= ssim(imagem,imagem_resultante,multichannel=True)
                with open (".../Segmentacao/Grafo/Resultados_Metricas/"+str(pasta)+"/ssim/"+str(arquivo)+".txt","w") as arq2:36arq2.write(str(resultado_ssim))
                
            if pasta == "Girassol 5_Rec":
                sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                ponto_s = (1805,1340)
                ponto_t = (1760,440) 
                imagem = cv2.equalizeHist(sementes)
                media_s = seg.mediaCinza(ponto_s, imagem, 5, 5)
                media_t = seg.mediaCinza(ponto_t, imagem, 5, 5)
                G = seg.criaDigrafo(imagem, media_s, media_t, ponto_s, ponto_t,"s", "t", seg.modulo )
                flow_dict, nos_dict = seg.ff(G, "s")              
                imagem_resultado = seg.segmentar(nos_dict, imagem.shape[0], imagem.shape[1] )
                imagem_normalizada = ((imagem_resultado  - np.min(imagem_resultado )) / (np.max(imagem_resultado) - np.min(imagem_resultado))) * 256
                cv2.imwrite("C.../Segmentacao/Grafo/Imagens/"+str(pasta)+"/"+str(arquivo) + ".bmp", imagem_normalizada)
                imagem_resultante = imagem_normalizada * imagem
                resultado_ssim= ssim(imagem,imagem_resultante,multichannel=True)
                with open (".../Segmentacao/Grafo/Resultados_Metricas/"+str(pasta)+"/ssim/"+str(arquivo)+".txt","w") as arq2:36arq2.write(str(resultado_ssim))
   

            if pasta == "Mix_Sementes":
                sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                ponto_s = (1700,730)
                ponto_t = (250,1470) 
                imagem = cv2.equalizeHist(sementes)
                media_s = seg.mediaCinza(ponto_s, imagem, 5, 5)
                media_t = seg.mediaCinza(ponto_t, imagem, 5, 5)
                G = seg.criaDigrafo(imagem, media_s, media_t, ponto_s, ponto_t,"s", "t", seg.modulo )
                flow_dict, nos_dict = seg.ff(G, "s")              
                imagem_resultado = seg.segmentar(nos_dict, imagem.shape[0], imagem.shape[1] )
                imagem_normalizada = ((imagem_resultado  - np.min(imagem_resultado )) / (np.max(imagem_resultado) - np.min(imagem_resultado))) * 256
                cv2.imwrite("C.../Segmentacao/Grafo/Imagens/"+str(pasta)+"/"+str(arquivo) + ".bmp", imagem_normalizada)
                imagem_resultante = imagem_normalizada * imagem
                resultado_ssim= ssim(imagem,imagem_resultante,multichannel=True)
                with open (".../Segmentacao/Grafo/Resultados_Metricas/"+str(pasta)+"/ssim/"+str(arquivo)+".txt","w") as arq2:36arq2.write(str(resultado_ssim))

            if pasta == "Pinhao Manso 1_Rec":
                sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                ponto_s = (200,580)
                ponto_t = (1340,710) 
                imagem = cv2.equalizeHist(sementes)
                media_s = seg.mediaCinza(ponto_s, imagem, 5, 5)
                media_t = seg.mediaCinza(ponto_t, imagem, 5, 5)
                G = seg.criaDigrafo(imagem, media_s, media_t, ponto_s, ponto_t,"s", "t", seg.modulo )
                flow_dict, nos_dict = seg.ff(G, "s")              
                imagem_resultado = seg.segmentar(nos_dict, imagem.shape[0], imagem.shape[1] )
                imagem_normalizada = ((imagem_resultado  - np.min(imagem_resultado )) / (np.max(imagem_resultado) - np.min(imagem_resultado))) * 256
                cv2.imwrite("C.../Segmentacao/Grafo/Imagens/"+str(pasta)+"/"+str(arquivo) + ".bmp", imagem_normalizada)
                imagem_resultante = imagem_normalizada * imagem
                resultado_ssim= ssim(imagem,imagem_resultante,multichannel=True)
                with open (".../Segmentacao/Grafo/Resultados_Metricas/"+str(pasta)+"/ssim/"+str(arquivo)+".txt","w") as arq2:36arq2.write(str(resultado_ssim))
                
                 
            if pasta == "Pinhao Manso 2_Rec":
                sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                ponto_s = (780,1230)
                ponto_t = (250,1470) 
                imagem = cv2.equalizeHist(sementes)
                media_s = seg.mediaCinza(ponto_s, imagem, 5, 5)
                media_t = seg.mediaCinza(ponto_t, imagem, 5, 5)
                G = seg.criaDigrafo(imagem, media_s, media_t, ponto_s, ponto_t,"s", "t", seg.modulo )
                flow_dict, nos_dict = seg.ff(G, "s")              
                imagem_resultado = seg.segmentar(nos_dict, imagem.shape[0], imagem.shape[1] )
                imagem_normalizada = ((imagem_resultado  - np.min(imagem_resultado )) / (np.max(imagem_resultado) - np.min(imagem_resultado))) * 256
                cv2.imwrite("C.../Segmentacao/Grafo/Imagens/"+str(pasta)+"/"+str(arquivo) + ".bmp", imagem_normalizada)
                imagem_resultante = imagem_normalizada * imagem
                resultado_ssim= ssim(imagem,imagem_resultante,multichannel=True)
                with open (".../Segmentacao/Grafo/Resultados_Metricas/"+str(pasta)+"/ssim/"+str(arquivo)+".txt","w") as arq2:36arq2.write(str(resultado_ssim))
                
            if pasta == "Pinhao Manso 3_Rec":
                sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                ponto_s = (1010,910)
                ponto_t = (1060,250) 
                imagem = cv2.equalizeHist(sementes)
                media_s = seg.mediaCinza(ponto_s, imagem, 5, 5)
                media_t = seg.mediaCinza(ponto_t, imagem, 5, 5)
                G = seg.criaDigrafo(imagem, media_s, media_t, ponto_s, ponto_t,"s", "t", seg.modulo )
                flow_dict, nos_dict = seg.ff(G, "s")              
                imagem_resultado = seg.segmentar(nos_dict, imagem.shape[0], imagem.shape[1] )
                imagem_normalizada = ((imagem_resultado  - np.min(imagem_resultado )) / (np.max(imagem_resultado) - np.min(imagem_resultado))) * 256
                cv2.imwrite("C.../Segmentacao/Grafo/Imagens/"+str(pasta)+"/"+str(arquivo) + ".bmp", imagem_normalizada)
                imagem_resultante = imagem_normalizada * imagem
                resultado_ssim= ssim(imagem,imagem_resultante,multichannel=True)
                with open (".../Segmentacao/Grafo/Resultados_Metricas/"+str(pasta)+"/ssim/"+str(arquivo)+".txt","w") as arq2:36arq2.write(str(resultado_ssim))
            
            if pasta == "Pinhao Manso 4_Rec":
                sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                ponto_s = (1500,1410)
                ponto_t = (1380,790)
                imagem = cv2.equalizeHist(sementes)
                media_s = seg.mediaCinza(ponto_s, imagem, 5, 5)
                media_t = seg.mediaCinza(ponto_t, imagem, 5, 5)
                G = seg.criaDigrafo(imagem, media_s, media_t, ponto_s, ponto_t,"s", "t", seg.modulo )
                flow_dict, nos_dict = seg.ff(G, "s")              
                imagem_resultado = seg.segmentar(nos_dict, imagem.shape[0], imagem.shape[1] )
                imagem_normalizada = ((imagem_resultado  - np.min(imagem_resultado )) / (np.max(imagem_resultado) - np.min(imagem_resultado))) * 256
                cv2.imwrite("C.../Segmentacao/Grafo/Imagens/"+str(pasta)+"/"+str(arquivo) + ".bmp", imagem_normalizada)
                imagem_resultante = imagem_normalizada * imagem
                resultado_ssim= ssim(imagem,imagem_resultante,multichannel=True)
                with open (".../Segmentacao/Grafo/Resultados_Metricas/"+str(pasta)+"/ssim/"+str(arquivo)+".txt","w") as arq2:36arq2.write(str(resultado_ssim))
                
            if pasta == "Pinhao Manso 5_Rec":
                sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                ponto_s = (1450,1400)
                ponto_t = (590,1510) 
                imagem = cv2.equalizeHist(sementes)
                media_s = seg.mediaCinza(ponto_s, imagem, 5, 5)
                media_t = seg.mediaCinza(ponto_t, imagem, 5, 5)
                G = seg.criaDigrafo(imagem, media_s, media_t, ponto_s, ponto_t,"s", "t", seg.modulo )
                flow_dict, nos_dict = seg.ff(G, "s")              
                imagem_resultado = seg.segmentar(nos_dict, imagem.shape[0], imagem.shape[1] )
                imagem_normalizada = ((imagem_resultado  - np.min(imagem_resultado )) / (np.max(imagem_resultado) - np.min(imagem_resultado))) * 256
                cv2.imwrite("C.../Segmentacao/Grafo/Imagens/"+str(pasta)+"/"+str(arquivo) + ".bmp", imagem_normalizada)
                imagem_resultante = imagem_normalizada * imagem
                resultado_ssim= ssim(imagem,imagem_resultante,multichannel=True)
                with open (".../Segmentacao/Grafo/Resultados_Metricas/"+str(pasta)+"/ssim/"+str(arquivo)+".txt","w") as arq2:36arq2.write(str(resultado_ssim))
           
            if pasta == "Soja 1_Rec":
                sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                ponto_s = (370,1480)
                ponto_t = (1500,820) 
                imagem = cv2.equalizeHist(sementes)
                media_s = seg.mediaCinza(ponto_s, imagem, 5, 5)
                media_t = seg.mediaCinza(ponto_t, imagem, 5, 5)
                G = seg.criaDigrafo(imagem, media_s, media_t, ponto_s, ponto_t,"s", "t", seg.modulo )
                flow_dict, nos_dict = seg.ff(G, "s")              
                imagem_resultado = seg.segmentar(nos_dict, imagem.shape[0], imagem.shape[1] )
                imagem_normalizada = ((imagem_resultado  - np.min(imagem_resultado )) / (np.max(imagem_resultado) - np.min(imagem_resultado))) * 256
                cv2.imwrite("C.../Segmentacao/Grafo/Imagens/"+str(pasta)+"/"+str(arquivo) + ".bmp", imagem_normalizada)
                imagem_resultante = imagem_normalizada * imagem
                resultado_ssim= ssim(imagem,imagem_resultante,multichannel=True)
                with open (".../Segmentacao/Grafo/Resultados_Metricas/"+str(pasta)+"/ssim/"+str(arquivo)+".txt","w") as arq2:36arq2.write(str(resultado_ssim))
                
            if pasta == "Soja 2_Rec":
                sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                ponto_s = (370,1480)
                ponto_t = (440,650) 
                imagem = cv2.equalizeHist(sementes)
                media_s = seg.mediaCinza(ponto_s, imagem, 5, 5)
                media_t = seg.mediaCinza(ponto_t, imagem, 5, 5)
                G = seg.criaDigrafo(imagem, media_s, media_t, ponto_s, ponto_t,"s", "t", seg.modulo )
                flow_dict, nos_dict = seg.ff(G, "s")              
                imagem_resultado = seg.segmentar(nos_dict, imagem.shape[0], imagem.shape[1] )
                imagem_normalizada = ((imagem_resultado  - np.min(imagem_resultado )) / (np.max(imagem_resultado) - np.min(imagem_resultado))) * 256
                cv2.imwrite("C.../Segmentacao/Grafo/Imagens/"+str(pasta)+"/"+str(arquivo) + ".bmp", imagem_normalizada)
                imagem_resultante = imagem_normalizada * imagem
                resultado_ssim= ssim(imagem,imagem_resultante,multichannel=True)
                with open (".../Segmentacao/Grafo/Resultados_Metricas/"+str(pasta)+"/ssim/"+str(arquivo)+".txt","w") as arq2:
                    arq2.write(str(resultado_ssim))
                
            if pasta == "Soja 3_Rec":
                sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                ponto_s = (44,65)
                ponto_t = (101,33)
                imagem = cv2.equalizeHist(sementes)
                media_s = seg.mediaCinza(ponto_s, imagem, 5, 5)
                media_t = seg.mediaCinza(ponto_t, imagem, 5, 5)
                G = seg.criaDigrafo(imagem, media_s, media_t, ponto_s, ponto_t,"s", "t", seg.modulo )
                flow_dict, nos_dict = seg.ff(G, "s")              
                imagem_resultado = seg.segmentar(nos_dict, imagem.shape[0], imagem.shape[1] )
                imagem_normalizada = ((imagem_resultado  - np.min(imagem_resultado )) / (np.max(imagem_resultado) - np.min(imagem_resultado))) * 256
                cv2.imwrite("C.../Segmentacao/Grafo/Imagens/"+str(pasta)+"/"+str(arquivo) + ".bmp", imagem_normalizada)
                imagem_resultante = imagem_normalizada * imagem
                resultado_ssim= ssim(imagem,imagem_resultante,multichannel=True)
                with open (".../Segmentacao/Grafo/Resultados_Metricas/"+str(pasta)+"/ssim/"+str(arquivo)+".txt","w") as arq2:
                    arq2.write(str(resultado_ssim))
                    
            if pasta == "Soja 4_Rec":
                sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                ponto_s = (380,1250)
                ponto_t = (1360,1730) 
                imagem = cv2.equalizeHist(sementes)
                media_s = seg.mediaCinza(ponto_s, imagem, 5, 5)
                media_t = seg.mediaCinza(ponto_t, imagem, 5, 5)
                G = seg.criaDigrafo(imagem, media_s, media_t, ponto_s, ponto_t,"s", "t", seg.modulo )
                flow_dict, nos_dict = seg.ff(G, "s")              
                imagem_resultado = seg.segmentar(nos_dict, imagem.shape[0], imagem.shape[1] )
                imagem_normalizada = ((imagem_resultado  - np.min(imagem_resultado )) / (np.max(imagem_resultado) - np.min(imagem_resultado))) * 256
                cv2.imwrite("C.../Segmentacao/Grafo/Imagens/"+str(pasta)+"/"+str(arquivo) + ".bmp", imagem_normalizada)
                imagem_resultante = imagem_normalizada * imagem
                resultado_ssim= ssim(imagem,imagem_resultante,multichannel=True)
                with open (".../Segmentacao/Grafo/Resultados_Metricas/"+str(pasta)+"/ssim/"+str(arquivo)+".txt","w") as arq2:
                    arq2.write(str(resultado_ssim))
                
            if pasta == "Soja 5_Rec":
                sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                ponto_s = (1060,1920)
                ponto_t = (960,750) 
                imagem = cv2.equalizeHist(sementes)
                media_s = seg.mediaCinza(ponto_s, imagem, 5, 5)
                media_t = seg.mediaCinza(ponto_t, imagem, 5, 5)
                G = seg.criaDigrafo(imagem, media_s, media_t, ponto_s, ponto_t,"s", "t", seg.modulo )
                flow_dict, nos_dict = seg.ff(G, "s")              
                imagem_resultado = seg.segmentar(nos_dict, imagem.shape[0], imagem.shape[1] )
                imagem_normalizada = ((imagem_resultado  - np.min(imagem_resultado )) / (np.max(imagem_resultado) - np.min(imagem_resultado))) * 256
                cv2.imwrite("C.../Segmentacao/Grafo/Imagens/"+str(pasta)+"/"+str(arquivo) + ".bmp", imagem_normalizada)
                imagem_resultante = imagem_normalizada * imagem
                resultado_ssim= ssim(imagem,imagem_resultante,multichannel=True)
                with open (".../Segmentacao/Grafo/Resultados_Metricas/"+str(pasta)+"/ssim/"+str(arquivo)+".txt","w") as arq2:
                    arq2.write(str(resultado_ssim))
                
    return soma
            

ler = grafo()

cv2.waitKey(0)
cv2.destroyAllWindows()