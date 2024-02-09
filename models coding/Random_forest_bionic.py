import numpy as np
import matplotlib as plt
import pandas as pd
import os
# print(os.listdir("../input"))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error,r2_score,classification_report

#Reading the data and adding them up
df0 = pd.read_csv('0.csv',header=None)
df1 = pd.read_csv('1.csv',header=None)
df2 = pd.read_csv('2.csv',header=None)
df3 = pd.read_csv('3.csv',header=None)
df = pd.concat([df0, df1, df2, df3])
# print(df)

#setting the values x and y traversing through
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

#Splitting and scaling the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# print(X_train)

#training and testing the data
classifier = RandomForestClassifier(n_estimators=20, random_state=0, criterion='entropy')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#Models credibility parameters
cm = confusion_matrix(y_test, y_pred)
accuracies = cross_val_score(estimator=classifier, X=X, y=Y, cv=10)
# if working with a lot of data, you can set n_jobs to -1
accuracies.mean()
accuracies.std()

#printing the accuracies and other testing parameters
print("mean accuracies=",accuracies.mean())
print("standard accuracies=",accuracies.std())
print("Test set classification rate: {}".format(np.mean(y_pred == y_test)))
print('\n')
print('Classification Report:\n',classification_report(y_test,y_pred))

rmse=mean_squared_error(y_test,y_pred, squared=False)
r2=r2_score(y_test,y_pred)
print('Root Mean Squared Error: ', rmse)
print('R2 Score :\t', r2)