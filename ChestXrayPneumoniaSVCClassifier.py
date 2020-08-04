import cv2
import os
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis

train_dir = 'G:\\Jupyter\\Chest-xray\\chest_xray\\train\\'
test_dir = 'G:\\Jupyter\\Chest-xray\\chest_xray\\test\\'

categories = ['NORMAL', 'PNEUMONIA']

train_data = []
train_label = []
IMG_SIZE = 100


def label_image(img):
    if('virus' in img): return 'virus'
    elif('bacteria' in img): return 'bacteria'
    else: return 'normal'



def create_train_data():
    for category in categories:
        folder = os.path.join(train_dir, category)
        for img in tqdm(os.listdir(folder)):
            label = label_image(img)
            path = os.path.join(folder,img)
            img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
            train_data.append(np.array(img).ravel())
            train_label.append(label)


test_data = []
test_label = []

def process_test_data():
    test_data_set = []
    for category in categories:
        folder = os.path.join(test_dir, category)
        for img in tqdm(os.listdir(folder)):
            label = label_image(img)
            path = os.path.join(folder, img)
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
            test_data.append(np.array(img).ravel())
            test_label.append(label)




create_train_data()
process_test_data()

model = SVC()
#model = KNeighborsClassifier(3)
#model = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
#model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
#model =  LinearDiscriminantAnalysis()
model.fit(train_data,train_label)

p = model.predict(test_data)

count=0
for i in range(0,len(p)) :
    print(p[i] + " : "+ test_label[i])
    if p[i] == test_label[i]:
        count += 1

print("Accuracy is : " + str(count/len(p) * 100))
