import numpy as np
import cv2
import os
from random import shuffle
from tqdm import  tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


train_dir = 'G:\\Jupyter\\DogCat\\train_data'
test_dir = 'G:\\Jupyter\\DogCat\\test_data'

IMAGE_SIZE = 50
LR = 1e-3

def label_image(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat':
        return 'cat'
    elif word_label == 'dog':
        return 'dog'


train_data = []
train_label = []

'''
img = cv2.resize(cv2.imread('G:\\Jupyter\\DogCat\\train_data\\cat.1.jpg',cv2.IMREAD_COLOR),(IMAGE_SIZE,IMAGE_SIZE))
nsamples, nx, ny = img.shape
print(str(nsamples) +" "+str(nx) +" "+str(ny))
d2_image = img.reshape((nsamples,nx*ny))
print(d2_image.shape)
print(type(d2_image))
print(d2_image.shape)
'''



def create_train_data():
    train_data_set = []
    for img in tqdm(os.listdir(train_dir)) :
        if img.split('.')[-1] == 'jpg':
            label = label_image(img)
            path = os.path.join(train_dir,img)
            img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE,IMAGE_SIZE))
            img_arr = np.array(img)
            train_data.append(img_arr.ravel())
            train_label.append(label)

test_data = []
test_label = []


def process_test_data():
    test_data_set = []
    for img in tqdm(os.listdir(test_dir)):
        if img.split('.')[-1] == 'jpg':
            path = os.path.join(test_dir,img)
            label = label_image(img)
            img_num = img.split('.')[0]
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE))
            img_arr = np.array(img)
            test_data.append(img_arr.ravel())
            test_label.append(label)


create_train_data()
process_test_data()

print(train_label[0])
print(train_data[0])
print('Axis 0 size : ', np.size(train_data[0], 0))


model = SVC()
model.fit(train_data, train_label)

p = model.predict(test_data)
count=0
for i in range(0,len(p)) :
    print(p[i] + " : "+ test_label[i])
    if p[i] == test_label[i]:
        count += 1

print("Accuracy is : "+ str(count/len(p) * 100))

print(len(test_data))













