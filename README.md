# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SANDHIYA M
RegisterNumber:  212224220086

```
```
import pandas as pd
from google.colab import files
import pandas as pd

# Upload the CSV
uploaded = files.upload()

# Read into a DataFrame (replace 'filename.csv' with your actual file name)
data = pd.read_csv("Placement_Data.csv")
data.head()


data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )
data1["status"]=le.fit_transform(data1["status"])
print(data1,"\n")

x=data1.iloc[:,:-1]
print(x,"\n")

y=data1["status"]
print(y,"\n")

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred,"\n")

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy,"\n")

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion,"\n")

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1,"\n")

print(lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]))
```

## Output:

### TOP 5 ELEMENTS

<img width="1350" height="661" alt="image" src="https://github.com/user-attachments/assets/89500cde-ee08-486c-b373-7d14881a5708" />


### PRINT DATA

<img width="843" height="279" alt="image" src="https://github.com/user-attachments/assets/9e4bf844-e6bb-4ec2-bb70-0d8a7fd0a0ba" />


### DATA_STATUS

<img width="573" height="310" alt="image" src="https://github.com/user-attachments/assets/5049672a-a2cf-47fe-9578-6574e6a181f9" />


### Y_PREDICTION ARRAY

<img width="542" height="272" alt="image" src="https://github.com/user-attachments/assets/65af023d-873d-4481-aeb5-ab9a0e87527d" />


### CONFUSION ARRAY

<img width="917" height="77" alt="image" src="https://github.com/user-attachments/assets/f1cd4a02-af71-4be5-b624-0aa8e2c988c7" />

### ACCURACY VALUE

<img width="295" height="36" alt="image" src="https://github.com/user-attachments/assets/0bd05d64-bc4c-455e-8683-62d4e250eeef" />

### CLASSFICATION REPORT

<img width="746" height="198" alt="image" src="https://github.com/user-attachments/assets/4e750cd6-ab73-4dac-b56f-6a583040ceb9" />

### PREDICTION

<img width="152" height="27" alt="image" src="https://github.com/user-attachments/assets/4f7447dd-7210-4114-8156-399bc638a619" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
