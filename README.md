<H3>ENTER YOUR NAME : Syed Abbu Rehan</H3>
<H3>ENTER YOUR REGISTER NO : 212223240165</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 10-03-2025</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
## Import Libraries
```
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
```
## Read the dataset
```
df=pd.read_csv("Churn_Modelling.csv")
print(df)
```
## Values of X and Y
```
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(X)
print(y)
```
## Check for outliers
```
df.describe()
```
## Check the missing data
```
print(df.isnull().sum())
df.fillna(df.mean().round(1), inplace=True)
print(df.isnull().sum())
y = df.iloc[:, -1].values
print(y)
```
## Checking for duplicates
```
df.duplicated()
```
## Dropping string value from dataset
```
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
```
## Normalize the dataset
```
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
```
## Training and testing the model
```
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```
## OUTPUT:
## Reading Data
![image](https://github.com/user-attachments/assets/9f0ccc78-1ef5-4fc9-9a3c-9bbe2080867f)
## Values of X and Y
![image](https://github.com/user-attachments/assets/2377e317-0c59-4e8f-9617-3d259bce6528)
## Check for outliers
![image](https://github.com/user-attachments/assets/0abacebf-b5f5-483e-9710-a40dffef20a5)
## Outliers
![image](https://github.com/user-attachments/assets/1e45f1cb-ed30-4829-8582-08469565023b)
## Check the missing data
![image](https://github.com/user-attachments/assets/1dcd233e-1ddc-4f06-9fc0-fa9881068163)
## Checking for duplicates
![image](https://github.com/user-attachments/assets/e405e9af-cfff-4d87-9002-1dc6920981af)
## Dropping string value from dataset
![image](https://github.com/user-attachments/assets/658235b6-bd04-4e11-9b16-15c39ffbedfa)
## Normalize the dataset
![image](https://github.com/user-attachments/assets/23818dff-44a9-402b-a956-53348ef36152)
## Training and testing the model
![image](https://github.com/user-attachments/assets/580c19b1-1929-4a35-8e02-37c2c807b7fb)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


