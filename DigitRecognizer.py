import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

train_data = pd.read_csv("G:\\Jupyter\\digit-recognizer\\train.csv")
test_data = pd.read_csv("G:\\Jupyter\\digit-recognizer\\test.csv")


# Training Data set
xtrain = train_data.iloc[0:21000,1:]
# Label train data
train_label = train_data.iloc[0:21000,0]

# Test from train data set
xtest = train_data.iloc[21000:,1:]
test_label = train_data.iloc[21000:, 0]

decision = SVC()
decision.fit(xtrain,train_label)
p = decision.predict(xtest)

count=0
for i in range(0,21000):
    if p[i] == test_label.iloc[i]:
        count += 1
print("Accuracy with SVC is : "+ str((count/21000)*100))

# Test Data set
xtest = test_data.iloc[0:, 0:]
print(len(xtest))

#Train Model with Decision Tree
#decision = SVC()
#decision.fit(xtrain,train_label)
p1 = decision.predict(xtest)

print("Preparing the csv file")

df = pd.DataFrame({
    "Label" : p1
})
df.insert(0,'ImageId', range(1,28001))
df.to_csv('G:\\Jupyter\\digit-recognizer\\submission.csv', index=False)