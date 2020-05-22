import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.svm import SVC

def extracao_classificacao():
    raiz= ".../imagem/"
    ids = [] 
    vectors = []
    hog_features = []
    hu_features = []
    for pasta in os.listdir(raiz):
        arquivos = os.listdir(raiz + pasta)
        for arquivo in arquivos:
            sementes = cv2.imread(raiz + pasta + "/" + arquivo)
            if pasta == "Girassol":                     
                sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)  
                imagem = cv2.resize(sementes,(256,256))
                features, hog_image = hog(imagem, 
                          orientations = 9, 
                          pixels_per_cell = (16, 16), 
                          cells_per_block = (2, 2), 
                          transform_sqrt = False, 
                          visualize = True, 
                          feature_vector = True)
                hog_features.append(features)
                huMoments = cv2.HuMoments(cv2.moments(imagem))
                hu_features.append(huMoments)      
                ids.append(pasta)
                a = np.append(features,huMoments)
                #a = [item[0] for item in a]
                vectors.append(a)
                
            if pasta == "Girassol_defeituosa":                     
                sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)  
                imagem = cv2.resize(sementes,(256,256))
                features, hog_image = hog(imagem, 
                          orientations = 9, 
                          pixels_per_cell = (16, 16), 
                          cells_per_block = (2, 2), 
                          transform_sqrt = False, 
                          visualize = True, 
                          feature_vector = True)
                hog_features.append(features)
                huMoments = cv2.HuMoments(cv2.moments(imagem))
                hu_features.append(huMoments)     
                ids.append(pasta)
                a = np.append(features,huMoments)
                #a = [item[0] for item in a]
                vectors.append(a)
                print(vectors)
                
            if pasta == "Mix_Sementes":                     
                sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)  
                imagem = cv2.resize(sementes,(256,256))
                features, hog_image = hog(imagem, 
                          orientations = 9, 
                          pixels_per_cell = (16, 16), 
                          cells_per_block = (2, 2), 
                          transform_sqrt = False, 
                          visualize = True, 
                          feature_vector = True)
                hog_features.append(features)
                huMoments = cv2.HuMoments(cv2.moments(imagem))
                hu_features.append(huMoments.astype(np.uint8))     
                ids.append(pasta)
                a = np.append(features,huMoments)
                #a = [item[0] for item in a]
                vectors.append(a)
                print(vectors)
                
            if pasta == "Pinhao Manso":
                sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)  
                imagem = cv2.resize(sementes,(256,256))
                features, hog_image = hog(imagem, 
                          orientations = 9, 
                          pixels_per_cell = (16, 16), 
                          cells_per_block = (2, 2), 
                          transform_sqrt = False, 
                          visualize = True, 
                          feature_vector = True)
                hog_features.append(features)
                huMoments = cv2.HuMoments(cv2.moments(imagem))
                hu_features.append(huMoments)     
                ids.append(pasta)
                a = np.append(features,huMoments)
                #a = [item[0] for item in a]
                vectors.append(a)
                print(vectors)
                
            if pasta == "Pinhao Manso_defeituosa":
                sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)  
                imagem = cv2.resize(sementes,(256,256))
                features, hog_image = hog(imagem, 
                          orientations = 9, 
                          pixels_per_cell = (16, 16), 
                          cells_per_block = (2, 2), 
                          transform_sqrt = False, 
                          visualize = True, 
                          feature_vector = True)
                hog_features.append(features)
                huMoments = cv2.HuMoments(cv2.moments(imagem))
                hu_features.append(huMoments)      
                ids.append(pasta)
                a = np.append(features,huMoments)
                #a = [item[0] for item in a]
                vectors.append(a)
                print(vectors)
                
            if pasta == "Soja":
                sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)  
                imagem = cv2.resize(sementes,(256,256))
                features, hog_image = hog(imagem, 
                          orientations = 9, 
                          pixels_per_cell = (16, 16), 
                          cells_per_block = (2, 2), 
                          transform_sqrt = False, 
                          visualize = True, 
                          feature_vector = True)
                hog_features.append(features)
                huMoments = cv2.HuMoments(cv2.moments(imagem))
                hu_features.append(huMoments)      
                ids.append(pasta)
                a = np.append(features,huMoments)
                #a = [item[0] for item in a]
                vectors.append(a)
                print(vectors)
                
            if pasta == "Soja_defeituosa":
                sementes = cv2.cvtColor(sementes, cv2.COLOR_BGR2GRAY)  
                imagem = cv2.resize(sementes,(256,256))
                features, hog_image = hog(imagem, 
                          orientations = 9, 
                          pixels_per_cell = (16, 16), 
                          cells_per_block = (2, 2), 
                          transform_sqrt = False, 
                          visualize = True, 
                          feature_vector = True)
                hog_features.append(features)
                huMoments = cv2.HuMoments(cv2.moments(imagem))
                hu_features.append(huMoments)      
                ids.append(pasta)
                a = np.append(features,huMoments)
                #a = [item[0] for item in a]
                vectors.append(a)
                print(vectors)
    vetor_caracteristicas = np.vstack(vectors).astype(np.float64)
    return np.array(ids), vetor_caracteristicas

ids,vectors = extracao_classificacao()
pca = PCA().fit(vectors.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("numero de componentes ")
plt.ylabel("variancia  cumulativa");

pca = PCA(n_components=125)
a = pca.fit(vectors)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)

X_proj_train = pca.fit_transform(vectors)
X_proj_test = pca.fit_transform(vectors_teste)

# 70\% training and 30\% test
X_train, X_test, y_train, y_test = train_test_split( X_proj_train, ids, test_size=0.3, random_state=40)

pd.Series(y_train).value_counts()

gnb = GaussianNB()
y_pred_gauss = gnb.fit(X_train, y_train)
ynew = gnb.predict(X_test)
print("Valor do predict",ynew)
(X_test.shape[0], (y_test != y_pred_gauss).sum()))
print("Acuracia:",metrics.accuracy_score(y_test, ynew))
cm = confusion_matrix(y_test, ynew)
sns.heatmap(cm, annot=True, fmt="d")
print(classification_report(y_test, ynew))

#SVM
svc_model = LinearSVC()
svc_model.fit(X_train,y_train)
vetor_label_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, vetor_label_predict)
sns.heatmap(cm, annot=True, fmt="d")
print(classification_report(y_test, vetor_label_predict))
print("Acuracia:",metrics.accuracy_score(y_test, vetor_label_predict))
cv2.waitKey(0)
cv2.destroyAllWindows()