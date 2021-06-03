import numpy as np
# !pip install opencv-contrib-python==3.4.2.17
import cv2
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold   
import keras
from keras.datasets import cifar10
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics

def desSIFT(image):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image,None)
    #draw keypoints
    #import matplotlib.pyplot as plt		
    #img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
    #plt.imshow(img2),plt.show()
    return kp,des

def describeSURF( image):
    surf = cv2.xfeatures2d.SURF_create()
    # it is better to have this value between 300 and 500
#     surf.setHessianThreshold(400)
    kp, des = surf.detectAndCompute(image,None)
    return kp,des


def getDescriptors(images, labels_g) : 
    descriptors = []
    labels = []
    count = 0
    
    print (images.shape)
    for image in images : 
        print (image.shape)
        #Converting the image into grayscale         
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        #Re-sizing the image
        image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_AREA)
        kp, des = describeSURF(image)
        
        if des is not None : 
            print (des.shape)
            descriptors.append(des)
            labels.append(int(labels_g[count]))
        count += 1
            
    
    print (len(labels))
    
    descriptorsFin = descriptors[0]
    
    for descriptor in descriptors[1:]:
        if descriptor is not None:
            descriptorsFin = np.vstack((descriptorsFin, descriptor))
    
        
    return descriptorsFin, descriptors, labels

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


sift_des, descriptors, sift_labels = getDescriptors(np.concatenate((x_train, x_test), axis = 0), np.concatenate((y_train, y_test)))

k = 50

voc,  variance = kmeans((sift_des), k, 1)
print (np.concatenate((y_train, y_test)).shape)

print (len(sift_labels))
print (len(sift_des))

# Constructing a histogram of k clusters and number of images having those clusters
imageFeatures = np.zeros((len(sift_labels), k), "float32")
for i in range(len(sift_labels)):
    words, distance = vq(descriptors[i],voc)
    for w in words:
        imageFeatures[i][w] += 1
stdSlr = StandardScaler().fit(imageFeatures)
imageFeatures = stdSlr.transform(imageFeatures)


X_train, X_test, y_train, y_test = train_test_split(imageFeatures, sift_labels, test_size=0.1, random_state=4)

clf = cv2.ml.KNearest_create()
clf.train(X_train, cv2.ml.ROW_SAMPLE, np.asarray(y_train, dtype=np.float32))

ret, results, neighbours ,dist = clf.findNearest(X_test, k=10)

pred_label = []
for var in results:
    label = var
    pred_label.append(int(label))

print (y_test)
print (pred_label)
    
# Measuring the accuracies
metrics.accuracy_score(y_test, pred_label)