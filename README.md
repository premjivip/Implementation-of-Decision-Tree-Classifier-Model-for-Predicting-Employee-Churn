# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Step-1
Import the libraries and read the data frame using pandas.
### Step-2
Calculate the null values from dataframe and apply label encoder.
### Step-3
Apply decision tree classifier on the dataframe.
### Step-4
Obtain the value of accuracy and data prediction.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: premji p
RegisterNumber:212221043004
*/


import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
### data.head():
![image](https://github.com/Yogabharathi3/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118899387/7d7c513b-bf01-4d79-9e36-2c222befd70b)
### data.info():
![image](https://github.com/Yogabharathi3/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118899387/6bcb1e1b-fcbf-45b1-8698-c6a06b0db043)
### is null() and sum():
![image](https://github.com/Yogabharathi3/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118899387/c1282d28-6371-42dd-a305-4b1490d63048)
### data value counts():
![image](https://github.com/Yogabharathi3/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118899387/35e2ba9a-be1e-47f1-8ba4-7cf955928024)
### data.head() for salary:
![image](https://github.com/Yogabharathi3/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118899387/e9cee245-c5af-47c2-a129-99400a9d2982)
### x.head():
![image](https://github.com/Yogabharathi3/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118899387/76494346-2d77-4c80-a786-4b7e48c402d2)
### Accuracy value:
![image](https://github.com/Yogabharathi3/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118899387/53cf7929-b2e5-4565-b1eb-5d228daa87f5)
### Data prediction:
![image](https://github.com/Yogabharathi3/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118899387/ba86731c-efb9-467e-9009-8ae7bb47debc)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
