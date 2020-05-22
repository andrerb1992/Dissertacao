
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import os
import SimpleITK
import matplotlib.pyplot as plt
import SimpleITK as sitk


def show_image1(img, title=None):
    nda = sitk.GetArrayViewFromImage(img)
    #nda = np.transpose(nda, (1, 2, 0))
    #print(nda.shape)
    plt.imshow(nda, cmap="gray")   
    plt.axis("off")
    if(title):
        plt.title(title, size=10)
                   
    
    
def show_image(img,pasta, arquivo, title=None):
    nda = sitk.GetArrayViewFromImage(img)
    #nda = np.transpose(nda, (1, 2, 0))
    #print(nda.shape)
    plt.imshow(nda, cmap="gray")   
    plt.axis("off")
    if(title):
        plt.title(title, size=10)
    cv2.imwrite(".../Imagens/Segmentacao/Regiao/binario/"+str(pasta)+"/"+str(arquivo) + ".bmp",nda)               
                    
def transformar5(img):
    for i in range(img.shape[1]):
        for j in range(img.shape[1]):
            if img[i][j]==0:
                img[i][j]=0
            elif img[i][j]==1:
                img[i][j]=255
    return img

def transformar6(img):
    for i in range(img.shape[1]):
        for j in range(img.shape[1]):
            if img[i][j]==0:
                img[i][j]=0
            elif img[i][j]==1:
                img[i][j]=255
    return img


def imagem_segmentation():
    
    raiz =".../Imagens/.../" 
    for pasta in os.listdir(raiz):
        arquivos = os.listdir(raiz + pasta)
        for arquivo in arquivos:
            sementes = cv2.imread(raiz + pasta + "/" + arquivo)
            if pasta == "Girassol 1_Rec": 
                imagem = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                imagem_T1 = sitk.GetImageFromArray(imagem) 
                imagem_T1_255 = sitk.Cast(sitk.RescaleIntensity(imagem_T1), sitk.sitkUInt8)                   show_image(imagem_T1,pasta,arquivo, "Original Image")
                seed = (780,1480) #coluna x linha
                seg = sitk.Image(imagem_T1.GetSize(), sitk.sitkUInt8)
                seg.CopyInformation(imagem_T1)
                seg[seed] = 1
                seg = sitk.BinaryDilate(seg, 2)
                show_image(imagem_T1, pasta, arquivo, "Original Image")
                plt.scatter(seed[0], seed[1], color="red", s=200)
                seg = sitk.ConnectedThreshold(imagem_T1, seedList=[seed], lower=40, upper=100)
                show_image(seg, pasta, arquivo, "Region Growing") 
                    
            if pasta == "Girassol 2_Rec":
                imagem = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY) imagem_T1 = sitk.GetImageFromArray(imagem) 
                imagem_T1_255 = sitk.Cast(sitk.RescaleIntensity(imagem_T1), sitk.sitkUInt8)
                show_image(imagem_T1,pasta,arquivo, "Original Image")
                seed = (780,1480) #coluna x linha
                seg = sitk.Image(imagem_T1.GetSize(), sitk.sitkUInt8)
                seg.CopyInformation(imagem_T1)
                seg[seed] = 1
                seg = sitk.BinaryDilate(seg, 2)
                show_image(imagem_T1, pasta, arquivo, "Original Image")
                plt.scatter(seed[0], seed[1], color="red", s=200)
                seg = sitk.ConnectedThreshold(imagem_T1, seedList=[seed], lower=40, upper=100)
                show_image(seg, pasta, arquivo, "Region Growing")           
            if pasta == "Girassol 3_Rec": 
                imagem = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                imagem_T1 = sitk.GetImageFromArray(imagem) 
                imagem_T1_255 = sitk.Cast(sitk.RescaleIntensity(imagem_T1), sitk.sitkUInt8)
                show_image(imagem_T1,pasta,arquivo, "Original Image")
                seed = (1000,1200) #coluna x linha
                seg = sitk.Image(imagem_T1.GetSize(), sitk.sitkUInt8)
                seg.CopyInformation(imagem_T1)
                seg[seed] = 1
                seg = sitk.BinaryDilate(seg, 2)
                show_image(imagem_T1, pasta, arquivo, "Original Image")
                plt.scatter(seed[0], seed[1], color="red", s=200)
                seg = sitk.ConnectedThreshold(imagem_T1, seedList=[seed], lower=40, upper=100)
                show_image(seg, pasta, arquivo, "Region Growing")       
            if pasta == "Girassol 4_Rec": 
                imagem = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                imagem_T1 = sitk.GetImageFromArray(imagem) 
                imagem_T1_255 =
                sitk.Cast(sitk.RescaleIntensity(imagem_T1), sitk.sitkUInt8)
                show_image(imagem_T1,pasta,arquivo, "Original Image")
                seed = (780,1480) #coluna x linha
                seg = sitk.Image(imagem_T1.GetSize(), sitk.sitkUInt8)
                seg.CopyInformation(imagem_T1)
                seg[seed] = 1
                seg = sitk.BinaryDilate(seg, 2)
                show_image(imagem_T1, pasta, arquivo, "Original Image")
                plt.scatter(seed[0], seed[1], color="red", s=200)
                seg = sitk.ConnectedThreshold(imagem_T1, seedList=[seed], lower=40, upper=100)
                show_image(seg, pasta, arquivo, "Region Growing")
               
            if pasta == "Girassol 5_Rec":
                imagem = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                imagem_T1 = sitk.GetImageFromArray(imagem)
                imagem_T1_255 = sitk.Cast(sitk.RescaleIntensity(imagem_T1), sitk.sitkUInt8)
                show_image(imagem_T1,pasta,arquivo, "Original Image")
                seed = (400,1700) #coluna x linha
                seg = sitk.Image(imagem_T1.GetSize(), sitk.sitkUInt8)
                seg.CopyInformation(imagem_T1)
                seg[seed] = 1
                seg = sitk.BinaryDilate(seg, 2)
                show_image(imagem_T1, pasta, arquivo, "Original Image")
                plt.scatter(seed[0], seed[1], color="red", s=200)
                seg = sitk.ConnectedThreshold(imagem_T1, seedList=[seed], lower=40, upper=100)
                show_image(seg, pasta, arquivo, "Region Growing") 
                    
            if pasta == "Mix_Sementes": 
                imagem = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                imagem_T1 = sitk.GetImageFromArray(imagem)
                imagem_T1_255 = sitk.Cast(sitk.RescaleIntensity(imagem_T1), sitk.sitkUInt8) 
                show_image(imagem_T1,pasta,arquivo, "Original Image")
                seed = (2500,1700) #coluna x linha
                seg = sitk.Image(imagem_T1.GetSize(), sitk.sitkUInt8)                    seg.CopyInformation(imagem_T1)                    seg[seed] = 1
                seg = sitk.BinaryDilate(seg, 2)                    show_image(imagem_T1, pasta, arquivo, "Original Image")
                plt.scatter(seed[0], seed[1], color="red", s=200)
                seg = sitk.ConnectedThreshold(imagem_T1, seedList=[seed], lower=40, upper=100)
                show_image(seg, pasta, arquivo, "Region Growing")
                
            if pasta == "Pinhao Manso 1_Rec": 
                imagem = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                imagem_T1 = sitk.GetImageFromArray(imagem)
                imagem_T1_255 = sitk.Cast(sitk.RescaleIntensity(imagem_T1), sitk.sitkUInt8)
                show_image(imagem_T1,pasta,arquivo, "Original Image")
                seed = (1900,600) #coluna x linha                  seg = sitk.Image(imagem_T1.GetSize(), sitk.sitkUInt8)
                seg.CopyInformation(imagem_T1)                    seg[seed] = 1
                seg = sitk.BinaryDilate(seg, 2)
                show_image(imagem_T1, pasta, arquivo, "Original Image")
                plt.scatter(seed[0], seed[1], color="red", s=200)
                seg = sitk.ConnectedThreshold(imagem_T1, seedList=[seed], lower=40, upper=100)
                show_image(seg, pasta, arquivo, "Region Growing")
                    
                    
            if pasta == "Pinhao Manso 2_Rec": 
                imagem = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                imagem_T1 = sitk.GetImageFromArray(imagem)
                imagem_T1_255 = sitk.Cast(sitk.RescaleIntensity(imagem_T1), sitk.sitkUInt8)
                show_image(imagem_T1,pasta,arquivo, "Original Image")
                seed = (1500,2300) #coluna x linha
                seg = sitk.Image(imagem_T1.GetSize(), sitk.sitkUInt8)                    seg.CopyInformation(imagem_T1)                    seg[seed] = 1                    
                seg = sitk.BinaryDilate(seg, 2)                    show_image(imagem_T1, pasta, arquivo, "Original Image")
                plt.scatter(seed[0], seed[1], color="red", s=200)
                seg = sitk.ConnectedThreshold(imagem_T1, seedList=[seed], lower=40, upper=100)
                show_image(seg, pasta, arquivo, "Region Growing")
                    
                    
            if pasta == "Pinhao Manso 3_Rec": 
                imagem = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY) imagem_T1 = sitk.GetImageFromArray(imagem)
                imagem_T1_255 = sitk.Cast(sitk.RescaleIntensity(imagem_T1), sitk.sitkUInt8)                    show_image(imagem_T1,pasta,arquivo, "Original Image")                    
                seed = (780,1480) #coluna x linha
                seg = sitk.Image(imagem_T1.GetSize(), sitk.sitkUInt8)
                seg.CopyInformation(imagem_T1)                    seg[seed] = 1                    
                seg = sitk.BinaryDilate(seg, 2)                    show_image(imagem_T1, pasta, arquivo, "Original Image")
                plt.scatter(seed[0], seed[1], color="red", s=200)
                seg = sitk.ConnectedThreshold(imagem_T1, seedList=[seed], lower=40, upper=100)
                show_image(seg, pasta, arquivo, "Region Growing")
                    
                
            if pasta == "Pinhao Manso 4_Rec": 
                imagem = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY) imagem_T1 = sitk.GetImageFromArray(imagem) 
                imagem_T1_255 = sitk.Cast(sitk.RescaleIntensity(imagem_T1), sitk.sitkUInt8)                    show_image(imagem_T1,pasta,arquivo, "Original Image")                    
                seed = (1000,2400) #coluna x linha
                seg = sitk.Image(imagem_T1.GetSize(), sitk.sitkUInt8)
                seg.CopyInformation(imagem_T1)
                seg[seed] = 1
                seg = sitk.BinaryDilate(seg, 2)
                show_image(imagem_T1, pasta, arquivo, "Original Image")
                plt.scatter(seed[0], seed[1], color="red", s=200)
                seg = sitk.ConnectedThreshold(imagem_T1, seedList=[seed], lower=40, upper=100)
                show_image(seg, pasta, arquivo, "Region Growing")
                    
            if pasta == "Pinhao Manso 5_Rec":
                imagem = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)
                imagem_T1 = sitk.GetImageFromArray(imagem) 
                imagem_T1_255 = sitk.Cast(sitk.RescaleIntensity(imagem_T1), sitk.sitkUInt8)                    show_image(imagem_T1,pasta,arquivo, "Original Image")
                seed = (2000,2400) #coluna x linha
                seg = sitk.Image(imagem_T1.GetSize(), sitk.sitkUInt8)
                seg.CopyInformation(imagem_T1)
                seg[seed] = 1
                seg = sitk.BinaryDilate(seg, 2)
                show_image(imagem_T1, pasta, arquivo, "Original Image")
                plt.scatter(seed[0], seed[1], color="red", s=200)
                seg = sitk.ConnectedThreshold(imagem_T1, seedList=[seed], lower=40, upper=100)
                show_image(seg, pasta, arquivo, "Region Growing")
                    
            if pasta == "Soja 1_Rec": 
                imagem = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)    imagem_T1 = sitk.GetImageFromArray(im) 
                    imagem_T1_255 = sitk.Cast(sitk.RescaleIntensity(imagem_T1), sitk.sitkUInt8)
                    plt.figure(figsize=(5,5))
                    show_image(imagem_T1,pasta,arquivo, "Original Image")
                    seed = (600,1000) #coluna x linha
                    seg = sitk.Image(imagem_T1.GetSize(), sitk.sitkUInt8)
                    seg.CopyInformation(imagem_T1)
                    seg[seed] = 1
                    seg = sitk.BinaryDilate(seg, 2)
                    show_image(imagem_T1, pasta, arquivo, "Original Image")
                    plt.scatter(seed[0], seed[1], color="red", s=200)
                seg = sitk.ConnectedThreshold(imagem_T1, seedList=[seed], lower=40, upper=100)
                    show_image(seg, pasta, arquivo, "Region Growing") 
                    
            if pasta == "Soja 2_Rec":
                imagem = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY) imagem_T1 = sitk.GetImageFromArray(imagem) 
                imagem_T1_255 = sitk.Cast(sitk.RescaleIntensity(imagem_T1), sitk.sitkUInt8)
                show_image(imagem_T1,pasta,arquivo, "Original Image")
                seed = (400,1300) #coluna x linha
                seg = sitk.Image(imagem_T1.GetSize(), sitk.sitkUInt8)
                seg.CopyInformation(imagem_T1)
                seg[seed] = 1
                seg = sitk.BinaryDilate(seg, 2)
                show_image(imagem_T1, pasta, arquivo, "Original Image")
                plt.scatter(seed[0], seed[1], color="red", s=200)
                seg = sitk.ConnectedThreshold(imagem_T1, seedList=[seed], lower=40, upper=100)
                show_image(seg, pasta, arquivo, "Region Growing")
                 
            if pasta == "Soja 3_Rec": 
                imagem = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY) imagem_T1 = sitk.GetImageFromArray(imagem) 
                imagem_T1_255 = sitk.Cast(sitk.RescaleIntensity(imagem_T1), sitk.sitkUInt8)
                show_image(imagem_T1,pasta,arquivo, "Original Image")
                seed = (1500,2500) #coluna x linha
                seg = sitk.Image(imagem_T1.GetSize(), sitk.sitkUInt8)
                seg.CopyInformation(imagem_T1)
                seg[seed] = 1
                seg = sitk.BinaryDilate(seg, 2)
                show_image(imagem_T1, pasta, arquivo, "Original Image")
                plt.scatter(seed[0], seed[1], color="red", s=200)
                seg = sitk.ConnectedThreshold(imagem_T1, seedList=[seed], lower=40, upper=100)
                show_image(seg, pasta, arquivo, "Region Growing")      
            if pasta == "Soja 4_Rec":
                imagem = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY) imagem_T1 = sitk.GetImageFromArray(imagem)
                imagem_T1_255 = sitk.Cast(sitk.RescaleIntensity(imagem_T1), sitk.sitkUInt8)
                show_image(imagem_T1,pasta,arquivo, "Original Image")
                seed = (1500,2550) #coluna x linha
                seg = sitk.Image(imagem_T1.GetSize(), sitk.sitkUInt8)
                seg.CopyInformation(imagem_T1)
                seg[seed] = 1
                seg = sitk.BinaryDilate(seg, 2)
                show_image(imagem_T1, pasta, arquivo, "Original Image")
                plt.scatter(seed[0], seed[1], color="red", s=200)
                seg = sitk.ConnectedThreshold(imagem_T1, seedList=[seed], lower=40, upper=100)
                show_image(seg, pasta, arquivo, "Region Growing")                    
            if pasta == "Soja 5_Rec": 
                imagem = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY) imagem_T1 = sitk.GetImageFromArray(imagem)
                imagem_T1_255 = sitk.Cast(sitk.RescaleIntensity(imagem_T1), sitk.sitkUInt8)
                show_image(imagem_T1,pasta,arquivo, "Original Image")
                seed = (2600,2000) 
                seg = sitk.Image(imagem_T1.GetSize(), sitk.sitkUInt8)
                seg.CopyInformation(imagem_T1)
                seg[seed] = 1
                seg = sitk.BinaryDilate(seg, 2)
                show_image(imagem_T1, pasta, arquivo, "Original Image")
                plt.scatter(seed[0], seed[1], color="red", s=200)
                seg = sitk.ConnectedThreshold(imagem_T1, seedList=[seed], lower=40, upper=100)
                show_image(seg, pasta, arquivo, "Region Growing") 
    return pasta

resultado = imagem_segmentation()
print(resultado)

resultado1 = imagem_segmentation_transformation()
print(resultado1)

cv2.waitKey(0)
cv2.destroyAllWindows()