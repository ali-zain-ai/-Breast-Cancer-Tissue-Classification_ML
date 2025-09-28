import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

dataset = pd.read_csv(r"D:\AI_Python\Breast Tissue Model\Breast_Tissue.csv")
# the first 5 rows of dataset to the data
print("First 5 rows of dataset\n.......................................\n",dataset.head(10))

#It will give all the information of dataset
print("INFORMATION OF DATASET\n.......................................\n",dataset.info())

# If will show the shape of dataset
print("Shape of dataset\n.......................................\n",dataset.shape)

# It will show the discriptive statistics of dataset 
print("Descriptive statistics of dataset\n.......................................\n",dataset.describe())

# it will give all the null values of datset
print("This will show how many null values are in the dataset\n.......................................\n", dataset.isnull().sum())

# print(dataset['Class'].value_counts())

# It will convert the non-numaric column in numaric form 
Encoder = LabelEncoder()
dataset['Class'] = Encoder.fit_transform(dataset['Class'])

#Seperating labels and the features
X = dataset.drop(columns=['Class'], axis = 1)

print(X.info())

Y = (dataset['Class'])
# print(Y)

features = dataset[['I0','PA500','HFS','DA','Area','A.DA','Max.IP','DR','P' ]]
# print(features)

# now we are spliting the training and testing data
X_train, X_test, Y_train, Y_test = (train_test_split(X, Y, test_size=0.2,stratify=Y,random_state=42))
scaler = StandardScaler()
X_train_std =scaler.fit_transform(X_train)
X_test_std =scaler.transform(X_test)

print(X_test.shape)

# #Training the model 
Model = DecisionTreeClassifier(max_depth=8)
Model.fit(X_train_std, Y_train)


# # model prediction 
Prediction = Model.predict(X_test_std)
print(Prediction)

# it will convert our encoded values back into its real form.
Encode_prediction = Encoder.inverse_transform(Prediction)
print(Encode_prediction)

# #model evaluation 
# '''it will find the:
# Precision
# Recall
# F1-score
# Accuracy
# '''
accuracy = accuracy_score(Y_test, Prediction)
print(f"Model Accuracy: {accuracy*100:.2f}%")
print(classification_report(Y_test, Prediction))
print(confusion_matrix(Y_test, Prediction))
# import pickle as pk
# pk.dump(Model, open('breast.pickel', 'wb'))
