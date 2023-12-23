import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


cc = pd.read_csv("creditcard.csv")
#print(cc.info())
x = cc.iloc[:,1:30].values
y = cc.iloc[:,30].values
#print(cc.isnull().values.any())
#print(cc[cc.Class==1])


train_a,test_a,train_b,test_b=train_test_split(x,y,test_size=0.2)
stdsc = StandardScaler()
train_a = stdsc.fit_transform(train_a)
test_a = stdsc.transform(test_a)
dtc = DecisionTreeClassifier(criterion= 'entropy',random_state=0)
dtc.fit(train_a,train_b)
dtc_pred = dtc.predict(test_a)
#print(dtc_pred)

con_matrix = confusion_matrix(dtc_pred,test_b)
accuracy_dtc = ((con_matrix[0][0] + con_matrix[1][1])/con_matrix.sum())*100
error_dtc = ((con_matrix[0][1] + con_matrix[1][0])/con_matrix.sum())*100
specificity_dtc = (con_matrix[1][1]/(con_matrix[1][1]+con_matrix[0][1]))*100
sensitivity_dtc = (con_matrix[0][0]/(con_matrix[0][0]+con_matrix[1][0]))*100

print("Accuracy_Decision (DecisionTreeClassier) = ",accuracy_dtc)
print("Error_Rate_Decision (DecisionTreeClassier) = ",error_dtc)
print("Specificity_Decision (DecisionTreeClassier) = ",specificity_dtc)
print("Sensitivity_Decision (DecisionTreeClassier) = ",sensitivity_dtc)



svc = SVC(kernel= 'rbf',random_state= 0)
svc.fit(train_a,train_b)
pred_svc = svc.predict(test_a)

con_matrix_1 = confusion_matrix(pred_svc,test_b)
accuracy_svc = ((con_matrix_1[0][0] + con_matrix_1[1][1])/con_matrix_1.sum())*100
error_svc = ((con_matrix_1[0][1] + con_matrix_1[1][0])/con_matrix_1.sum())*100
specificity_svc = (con_matrix_1[1][1]/(con_matrix_1[1][1]+con_matrix_1[0][1]))*100
sensitivity_svc = (con_matrix_1[0][0]/(con_matrix_1[0][0]+con_matrix_1[1][0]))*100

print("Accuracy_Decision (SVC) = ",accuracy_svc)
print("Error_Rate_Decision (SVC) = ",error_svc)
print("Specificity_Decision (SVC) = ",specificity_svc)
print("Sensitivity_Decision (SVC) = ",sensitivity_svc)
