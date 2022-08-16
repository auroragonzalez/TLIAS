import git # sudo pip3 install gitpython
import os
import requests
import numpy as np
import cv2 # sudo pip3 install opencv-python
import copy
import scipy
from pdf2image import convert_from_path # sudo pip3 install pdf2image
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
from time import time
from skimage import io # sudo pip3 install scikit-image
from multiprocessing.pool import ThreadPool

def lsremote(url):
    remote_refs = {}
    g = git.cmd.Git()
    for ref in g.ls_remote(url).split('\n'):
        hash_ref_list = ref.split('\t')
        remote_refs[hash_ref_list[1]] = hash_ref_list[0]
    return remote_refs


def lsremote(url):
    remote_refs = {}
    g = git.cmd.Git()
    for ref in g.ls_remote(url).split('\n'):
        hash_ref_list = ref.split('\t')
        remote_refs[hash_ref_list[1]] = hash_ref_list[0]
    return remote_refs


refs = lsremote('https://bitbucket.org/aurorax/datangi')
lastCommit = refs['HEAD']

print(lastCommit)

## read data
infoBuildings = pd.read_csv('https://bitbucket.org/aurorax/datangi/raw/'+lastCommit+'/processed/study1-energybuildingsEnergyInfo.csv')

print(infoBuildings)

names = list(infoBuildings.sort_values('uid')['uid'])
namesDf = infoBuildings.sort_values('uid')['uid']
namesDf.to_csv('./names.csv', index=False)

names[0]

trainIds = pd.read_csv("trainInfo.csv", sep=",")
trainIds.columns = ['names', 'id']
trainNames = trainIds['names']
trainIds

# select test buildings (id)
testIds = pd.read_csv("testInfo.csv", sep=";")
testIds.columns = ['names', 'id']
testNames = testIds['names']
testIds
list(namesDf.iloc[trainIds['id']]) == list(trainIds['names']) #Compruebo que estan bien ordenados
list(namesDf.iloc[testIds['id']]) == list(testIds['names']) #Compruebo que estan bien ordenados

start = time() # Elapsed time: 2 minutes, 4.6515 seconds.

def get_urls():
    urls = []
    for i in range(len(trainNames)):
        urlConsumo = 'https://bitbucket.org/aurorax/datangi/raw/' + lastCommit + '/processed/study1-energy/' + trainNames[i] + '_heat.jpg' 
        urlTemperatura = 'https://bitbucket.org/aurorax/datangi/raw/' + lastCommit + '/processed/study1-temperature/t' + trainNames[i] + '_heat.jpg'
        urls.append(urlConsumo)
        urls.append(urlTemperatura)
    
    return urls


    
def image_downloader(img_url: str):
    dir = os.path.join('./Images3/')
    if not os.path.exists(dir):
        os.mkdir(dir)
    res = requests.get(img_url, stream=True)
    count = 1
    while res.status_code != 200 and count <= 5:
        res = requests.get(img_url, stream=True)
        count += 1
    i = Image.open(BytesIO(res.content))
    i.save('./Images3/' + os.path.basename(img_url))
    return f'Download complete: {os.path.basename(img_url)}'


def run_downloader(process:int, images_url:list):
    results = ThreadPool(process).imap_unordered(image_downloader, images_url)
    for r in results:
        print(r)


def load_images():
    images = []
    for i in range(len(trainNames)):
        inputImages = []
        outputImage = np.zeros((224, 224, 3), dtype = "uint8")
        image = cv2.imread('./Images3/'+trainNames[i]+'_heat.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 112))
        inputImages.append(image)
        image = cv2.imread('Images3/t'+trainNames[i]+'_heat.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 112))
        inputImages.append(image)
        outputImage[0:112, 0:224] = inputImages[0]
        outputImage[112:224, 0:224] = inputImages[1]
        images.append(outputImage)
    return np.array(images)


#images_url = get_urls() # only first time
#run_downloader(25, images_url) # only first time
mosaico = load_images()

elapsed_time = time() - start
minutos = int(elapsed_time/60)
segundos = elapsed_time%60
print("Elapsed time: %d minutes, %.4f seconds." % (minutos, segundos))

import keras
import sklearn
import tensorflow
def load_and_resize(images):
    images = []    
    for img in images:
        image = cv2.resize(image, (224,224))
        images.append(image)
    return images

   
def normalise_images(images):
    # Convert to numpy arrays y pasamos de tipo uint8 a float32
    images = np.array(images, dtype = np.float32)
    images /= 255
    return images

imgNormalise = normalise_images(mosaico)
plt.imshow(mosaico[168])
plt.imshow(imgNormalise[168])

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from keras.applications import vgg16
from keras.applications import vgg19
#from keras.applications import resnet50
from keras.applications import resnet
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import pairwise_distances

def create_vgg16():
    model = vgg16.VGG16(include_top = False, weights = "imagenet", input_shape = (224,224,3))
    return model

vgg16_model = create_vgg16()
vgg16_model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['acc', 'mse'])
def create_vgg19():
    model = vgg19.VGG19(include_top = False, weights = "imagenet", input_shape = (224,224,3))
    return model

vgg19_model = create_vgg19()
vgg19_model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['acc', 'mse'])
def create_ResNet50():
    model = resnet50.ResNet50(include_top = False, weights = "imagenet", input_shape = (224,224,3))
    return model

#resNet50_model = create_ResNet50()
resNet50_model = resnet.ResNet50()
resNet50_model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['acc', 'mse'])
def covnet_transform(covnet_model, raw_images):
    # Pass our training data through the network
    pred = covnet_model.predict(raw_images)
    # Flatten the array
    flat = pred.reshape(raw_images.shape[0], -1)
    return flat


# NOT RUN, results already saved and can be loaded in the following chunck
vgg16_output = covnet_transform(vgg16_model, imgNormalise)
print("VGG16 flattened output has {} features".format(vgg16_output.shape[1]))

vgg19_output = covnet_transform(vgg19_model, imgNormalise)
print("VGG19 flattened output has {} features".format(vgg19_output.shape[1]))

resnet50_output = covnet_transform(resNet50_model, imgNormalise)
print("ResNet50 flattened output has {} features".format(resnet50_output.shape[1]))
# Function that creates a PCA instance, fits it to the data and returns the instance
# Create PCA instances for each covnet output
vgg16_pca = create_fit_PCA(vgg16_output)
vgg19_pca = create_fit_PCA(vgg19_output)
resnet50_pca = create_fit_PCA(resnet50_output)
# PCA transformations of covnet outputs
vgg16_output_pca = vgg16_pca.transform(vgg16_output)
vgg19_output_pca = vgg19_pca.transform(vgg19_output)
resnet50_output_pca = resnet50_pca.transform(resnet50_output)
np.save('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/vgg16_output.npy'
     , vgg16_output)
np.save('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/vgg19_output.npy'
     , vgg19_output)
np.save('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/resnet50_output'
     , resnet50_output)
np.save('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/vgg16_output_pca.npy'
     , vgg16_output_pca)
np.save('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/vgg19_output_pca.npy'
     , vgg19_output_pca)
np.save('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/resnet50_output_pca'
     , resnet50_output_pca)


vgg16_output = np.load('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/vgg16_output.npy')
vgg19_output = np.load('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/vgg19_output.npy')
resnet50_output = np.load('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/resnet50_output.npy')
vgg16_output_pca = np.load('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/vgg16_output.npy')
vgg19_output_pca = np.load('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/vgg19_output.npy')
resnet50_output_pca = np.load('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/resnet50_output.npy')

def create_train_kmeans(data, number_of_clusters):
    # n_jobs is set to -1 to use all available CPU cores. This makes a big difference on an 8-core CPU
    # especially when the data size gets much bigger. #perfMatters
    k = KMeans(n_clusters = number_of_clusters, n_jobs = -1, random_state = 728)
    # Let's do some timings to see how long it takes to train.
    start = time()
    # Train it up
    k.fit(data)
    # Stop the timing 
    end = time()
    # And see how long that took
    print("Training took {} seconds".format(end-start))
    return k
    

def create_train_gmm(data, number_of_clusters):
   # g = GaussianMixture(n_components = number_of_clusters, covariance_type = "full", random_state = 728)
    g = GaussianMixture(n_components = number_of_clusters, 
                        covariance_type = "diag", random_state = 728, reg_covar =1e-5)
    start = time()
    g.fit(data)
    end = time()
    print("Training took {} seconds".format(end-start))    
    return g


nClus = 15 # Numero de clusters
# Let's pass the data into the algorithm and predict who lies in which cluster. 
# Since we're using the same data that we trained it on, this should give us the training results.

# Here we create and fit a KMeans model with the PCA outputs
print("KMeans (PCA): \n")

print("VGG16")
K_vgg16_pca = create_train_kmeans(vgg16_output_pca, nClus)

print("\nVGG19")
K_vgg19_pca = create_train_kmeans(vgg19_output_pca, nClus)

print("\nResNet50")
K_resnet50_pca = create_train_kmeans(resnet50_output_pca, nClus)

# Same for Gaussian Model
print("GMM (PCA): \n")

print("VGG16")
G_vgg16_pca = create_train_gmm(vgg16_output_pca, nClus)

print("\nVGG19")
G_vgg19_pca = create_train_gmm(vgg19_output_pca, nClus)
print("\nResNet50")
G_resnet50_pca = create_train_gmm(resnet50_output_pca, nClus)
# Let's also create models for the covnet outputs without PCA for comparison
print("KMeans: \n")

print("VGG16:")
K_vgg16 = create_train_kmeans(vgg16_output, nClus)

print("\nVGG19:")
K_vgg19 = create_train_kmeans(vgg19_output, nClus)

print("\nResNet50:")
K_resnet50 = create_train_kmeans(resnet50_output, nClus)


#Calculo de las distancias de todos los elementos del conjunto train con sus centroides y me quedo con el minimo
clusterings = [K_vgg16, K_vgg16_pca, K_vgg19, K_vgg19_pca, K_resnet50, K_resnet50_pca]
clusteringsS = ['K_vgg16', 'K_vgg16_pca', 'K_vgg19', 'K_vgg19_pca', 'K_resnet50', 'K_resnet50_pca']
outputs = [vgg16_output, vgg16_output_pca, vgg19_output,vgg19_output_pca, resnet50_output, resnet50_output_pca]
for i in range(len(clusterings)):
    centroids = clusterings[i].cluster_centers_
    closest,_ = pairwise_distances_argmin_min(centroids, outputs[i])
    pd.DataFrame(closest, columns=['closest']).to_csv('./'+str(nClus)+'/closest_'+clusteringsS[i]+'_trainnClus'+str(nClus)+'.csv', index=False)
    _,dist = pairwise_distances_argmin_min(outputs[i], centroids)
    labelsAndDists = pd.DataFrame(dist)
    labelsAndDists['cluster'] = K_vgg16.labels_
    labelsAndDists.to_csv('./'+str(nClus)+'/labelsAndDist_'+clusteringsS[i]+'_train.csv', index=False)    


#Calculo de las distancias de todos los elementos del conjunto train con sus centroides y me quedo con el minimo
clusterings = [K_vgg16, K_vgg16_pca, K_vgg19, K_vgg19_pca, K_resnet50, K_resnet50_pca]
clusteringsS = ['K_vgg16', 'K_vgg16_pca', 'K_vgg19', 'K_vgg19_pca', 'K_resnet50', 'K_resnet50_pca']
outputs = [vgg16_output, vgg16_output_pca, vgg19_output,vgg19_output_pca, resnet50_output, resnet50_output_pca]
for i in range(len(clusterings)):
    centroids = clusterings[i].cluster_centers_
    closest,_ = pairwise_distances_argmin_min(centroids, outputs[i])
    pd.DataFrame(closest, columns=['closest']).to_csv('./'+str(nClus)+'/closest_'+clusteringsS[i]+'_trainnClus'+str(nClus)+'.csv', index=False)
    _,dist = pairwise_distances_argmin_min(outputs[i], centroids)
    labelsAndDists = pd.DataFrame(dist)
    labelsAndDists['cluster'] = K_vgg16.labels_
    labelsAndDists.to_csv('./'+str(nClus)+'/labelsAndDist_'+clusteringsS[i]+'_train.csv', index=False)

from numpy.random import standard_normal
from scipy.linalg import cholesky


X = vgg19_output_pca
centers = np.empty(shape=(G_vgg19_pca.n_components, X.shape[1]))
for i in range(G_vgg19_pca.n_components):
    cov_matrix = np.diag(G_vgg19_pca.covariances_[i])
    l = cholesky(cov_matrix, check_finite=False, overwrite_a=True)
    density = G_vgg19_pca.means_[i] + l.dot(standard_normal(len(G_vgg19_pca.means_[i])))
    centers[i, :] = X[np.argmax(density)]
    
    
clusterings = [G_vgg16_pca]
clusteringsS = ['G_vgg16_pca']
outputs = [vgg16_output_pca]

k=0
centers = np.empty(shape=(G_vgg16_pca.n_components, outputs[k].shape[1]))
for i in range(clusterings[k].n_components):
    density = scipy.stats.multivariate_normal(cov=clusterings[k].covariances_[i], mean=clusterings[k].means_[i]).logpdf(X)
    centers[i, :] = X[np.argmax(density)]
    
    
closest,_ = pairwise_distances_argmin_min(centers, outputs[k])
pd.DataFrame(closest, columns=['closest']).to_csv('./'+str(nClus)+'/closest_'+clusteringsS[k]+'_trainnClus'+str(nClus)+'.csv', index=False)
label,dist = pairwise_distances_argmin_min(outputs[k], centers)
labelsAndDists = pd.DataFrame(dist)
labelsAndDists['cluster'] = label
labelsAndDists.to_csv('./'+str(nClus)+'/labelsAndDist_'+clusteringsS[i]+'_train.csv', index=False)
from numpy.random import standard_normal
from scipy.linalg import cholesky


X = vgg19_output_pca
centers = np.empty(shape=(G_vgg19_pca.n_components, X.shape[1]))
for i in range(G_vgg19_pca.n_components):
    cov_matrix = np.diag(G_vgg19_pca.covariances_[i])
    l = cholesky(cov_matrix, check_finite=False, overwrite_a=True)
    density = G_vgg19_pca.means_[i] + l.dot(standard_normal(len(G_vgg19_pca.means_[i])))
    centers[i, :] = X[np.argmax(density)]
closest,_ = pairwise_distances_argmin_min(centers, vgg19_output_pca)
closest
label,dist = pairwise_distances_argmin_min(vgg19_output_pca, centers)
labelsAndDists = pd.DataFrame(dist)
labelsAndDists['cluster'] = label
labelsAndDists.to_csv('labelsAndDist_vgg19gmm_train.csv', index=False)
labelsAndDists
X = resnet50_output_pca
centers = np.empty(shape=(G_resnet50_pca.n_components, X.shape[1]))
for i in range(G_resnet50_pca.n_components):
    density = scipy.stats.multivariate_normal(cov=G_resnet50_pca.covariances_[i], mean=G_resnet50_pca.means_[i]).logpdf(X)
    centers[i, :] = X[np.argmax(density)]
closest,_ = pairwise_distances_argmin_min(centers, resnet50_output_pca)
closest
label,dist = pairwise_distances_argmin_min(resnet50_output_pca, centers)
labelsAndDists = pd.DataFrame(dist)
labelsAndDists['cluster'] = label
labelsAndDists.to_csv('labelsAndDist_resnet50gmm_train.csv', index=False)
labelsAndDists

### PARTE DE TEST

refs = lsremote('https://bitbucket.org/aurorax/datangi3')
lastCommit = refs['HEAD']

print(lastCommit)

start = time() # Elapsed time: 0 minutes, 15.2394 seconds.

def get_urls():
    urls = []
    for i in range(len(testNames)):
        urlConsumo = 'https://bitbucket.org/aurorax/datangi3/raw/' + lastCommit + '/test-consumption/' + testNames[i] + '_heat.jpg' 
        urlTemperatura = 'https://bitbucket.org/aurorax/datangi3/raw/' + lastCommit + '/test-temperature/t' + testNames[i] + '_heat.jpg'
        urls.append(urlConsumo)
        urls.append(urlTemperatura)
    
    return urls
    
def image_downloader(img_url: str):
    res = requests.get(img_url, stream=True)
    count = 1
    while res.status_code != 200 and count <= 5:
        res = requests.get(img_url, stream=True)
        count += 1
    i = Image.open(BytesIO(res.content))
    i.save('./Images3/' + os.path.basename(img_url))
    return f'Download complete: {os.path.basename(img_url)}'


def run_downloader(process:int, images_url:list):
    results = ThreadPool(process).imap_unordered(image_downloader, images_url)
    for r in results:
        print(r)

def load_images():
    images = []
    for i in range(len(testNames)):
        inputImages = []
        outputImage = np.zeros((224, 224, 3), dtype = "uint8")
        image = cv2.imread('./Images3/'+testNames[i]+'_heat.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 112))
        inputImages.append(image)
        image = cv2.imread('./Images3/t'+testNames[i]+'_heat.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 112))
        inputImages.append(image)
        outputImage[0:112, 0:224] = inputImages[0]
        outputImage[112:224, 0:224] = inputImages[1]
        images.append(outputImage)
    return np.array(images)


images_url = get_urls()
run_downloader(25, images_url)
mosaicoTest = load_images()

elapsed_time = time() - start
minutos = int(elapsed_time/60)
segundos = elapsed_time%60
print("Elapsed time: %d minutes, %.4f seconds." % (minutos, segundos))


imgNormaliseTest = normalise_images(mosaicoTest)
plt.imshow(imgNormaliseTest[0])

vgg16_output_test = covnet_transform(vgg16_model, imgNormaliseTest)  # ¿no sería mejor aquí usar vgg16_model_?
print("VGG16 flattened output has {} features".format(vgg16_output_test.shape[1]))
vgg19_output_test = covnet_transform(vgg19_model, imgNormaliseTest)
print("VGG19 flattened output has {} features".format(vgg19_output_test.shape[1]))
resnet50_output_test = covnet_transform(resNet50_model, imgNormaliseTest)
print("ResNet50 flattened output has {} features".format(resnet50_output_test.shape[1]))

# Create PCA instances for each covnet output
vgg16_pca_test = create_fit_PCA(vgg16_output_test)
vgg19_pca_test = create_fit_PCA(vgg19_output_test)
resnet50_pca_test = create_fit_PCA(resnet50_output_test)
# PCA transformations of covnet outputs
vgg16_output_pca_test = vgg16_pca_test.transform(vgg16_output_test)
vgg19_output_pca_test = vgg19_pca_test.transform(vgg19_output_test)
resnet50_output_pca_test = resnet50_pca_test.transform(resnet50_output_test)
np.save('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/vgg16_output_pca_test.npy'
     , vgg16_output_pca_test)
np.save('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/vgg19_output_pca_test.npy'
     , vgg19_output_pca_test)
np.save('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/resnet50_output_pca_test'
     , resnet50_output_pca_test)

np.save('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/vgg16_output_test.npy'
     , vgg16_output_test)
np.save('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/vgg19_output_test.npy'
     , vgg19_output_test)
np.save('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/resnet50_output_test'
     , resnet50_output_test)

vgg16_output_test = np.load('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/vgg16_output_test.npy')
vgg19_output_test = np.load('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/vgg19_output_test.npy')
resnet50_output_test = np.load('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/resnet50_output_test.npy')
vgg16_output_pca_test = np.load('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/vgg16_output_test.npy')
vgg19_output_pca_test = np.load('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/vgg19_output_test.npy')
resnet50_output_pca_test = np.load('/home/aurorax/Git_repos/postdoc/estanciaERAU/models/Final_image_clustering/Tareas/numpyStorage/resnet50_output_test.npy')

k_vgg16_pred_test = K_vgg16.predict(vgg16_output_test)
pred = pd.DataFrame(testIds['id'])
pred['cluster'] = k_vgg16_pred_test
pred.to_csv('./'+str(nClus)+'/k_vgg16_pred_test.csv', index=False)
pred

centroids = K_vgg16.cluster_centers_
label,dist = pairwise_distances_argmin_min(vgg16_output_test, centroids)
labelsAndDists = pd.DataFrame(dist)
labelsAndDists['cluster'] = label
labelsAndDists.to_csv('./'+str(nClus)+'/labelsAndDist_vgg16kmeans_test.csv', index=False)
labelsAndDists

#Calculo de las distancias de todos los elementos del conjunto train con sus centroides y me quedo con el minimo
clusterings = [K_vgg16, K_vgg16_pca, K_vgg19, K_vgg19_pca, K_resnet50, K_resnet50_pca]
clusteringsS = ['K_vgg16', 'K_vgg16_pca', 'K_vgg19', 'K_vgg19_pca', 'K_resnet50', 'K_resnet50_pca']
outputs = [vgg16_output_test, vgg16_output_pca_test, vgg19_output_test,
           vgg19_output_pca_test, resnet50_output_test, resnet50_output_pca_test]
           
for i in range(len(clusterings)):
    pred_test = clusterings[i].predict(outputs[i])
    pred = pd.DataFrame(testIds['id'])
    pred['cluster'] = pred_test
    pred.to_csv('./'+str(nClus)+ '/' + clusteringsS[i]+ '_pred_test.csv', index=False)
    pred
    centroids = clusterings[i].cluster_centers_
    label,dist = pairwise_distances_argmin_min(outputs[i], centroids)
    labelsAndDists = pd.DataFrame(dist)
    labelsAndDists['cluster'] = label
    labelsAndDists.to_csv('./'+str(nClus)+'/labelsAndDist_'+clusteringsS[i]+'_test.csv', index=False)
    labelsAndDists

