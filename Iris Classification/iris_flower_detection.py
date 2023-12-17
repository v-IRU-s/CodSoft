import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

iris = pd.read_csv("iris.csv")
#print(iris)
train,test = train_test_split(iris,test_size=0.2)
#print(test)
train_a = train[['sepal_length','sepal_width','petal_length','petal_width']]
train_b = train.species
test_a = test[['sepal_length','sepal_width','petal_length','petal_width']]
test_b = test.species

model_svc = svm.SVC()
model_lr = LogisticRegression()
model_DTC = DecisionTreeClassifier()
model_svc.fit(train_a.values,train_b)
model_lr.fit(train_a.values,train_b)
model_DTC.fit(train_a.values,train_b)

pred_svc = model_svc.predict(test_a.values)
pred_lr = model_lr.predict(test_a.values)
pred_dtc = model_DTC.predict(test_a.values)

#print(pred)
print("Accuracy of Model (SVC) : ",(metrics.accuracy_score(pred_svc,test_b))*100,"%")
print("Accuracy of Model (Logistic Regression) : ",(metrics.accuracy_score(pred_lr,test_b))*100,"%")
print("Accuracy of Model (Decission Tree Classifier) : ",(metrics.accuracy_score(pred_dtc,test_b))*100,"%")

n=int(input("Enter no. of predictions to be performed : "))
for i in range(0,n):
    sepal_length = float(input("Enter sepal length : "))
    sepal_width = float(input("Enter sepal width : "))
    petal_length = float(input("Enter petal length : "))
    petal_width = float(input("Enter petal width : "))
    p = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    print("Predicted Flower using svc : ",model_svc.predict(p))
    print("Predicted Flower using LR : ",model_lr.predict(p))
    print("Predicted Flower using DTC : ",model_DTC.predict(p))