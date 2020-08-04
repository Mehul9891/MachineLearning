import numpy as np
import cv2
import os
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout , fully_connected
from tflearn.layers.estimator import regression


train_dir = 'G:\\Jupyter\\DogCat\\train_data'
test_dir = 'G:\\Jupyter\\DogCat\\test_data'

IMAGE_SIZE = 50
LR = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR,'2conv-basic')

def label_image(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat':
        return [1,0]
    elif word_label == 'dog':
        return [0,1]


def create_train_data():
    train_data_set = []
    for img in tqdm(os.listdir(train_dir)) :
        label = label_image(img)
        path = os.path.join(train_dir,img)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE,IMAGE_SIZE))
        train_data_set.append([np.array(img),np.array(label)])
    shuffle(train_data_set)
    np.save('train_data_set.npy',train_data_set)
    return train_data_set

def process_test_data():
    test_data_set = []
    for img in tqdm(os.listdir(test_dir)):
        path = os.path.join(test_data_set,img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE))
        test_data_set.append([np.array(img),img_num])
    np.save('test_data_set.npy',test_data_set)
    return test_data_set

train_data = create_train_data()


convnet = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print("Model loaded !!!")

train = train_data[:-500]
test = train_data[-500:]

X = np.array(i[0] for i in train).reshape(-1, IMAGE_SIZE,IMAGE_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array(i[0] for i in test).reshape(-1, IMAGE_SIZE,IMAGE_SIZE,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)






