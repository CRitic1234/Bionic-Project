import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score, accuracy_score

# Print out filenames for debugging
# print("CSV files in the current directory:", os.listdir())


# Reading the data and combining them
df0 = pd.read_csv('0.csv', header=None)
df1 = pd.read_csv('1.csv', header=None)
df2 = pd.read_csv('2.csv', header=None)
df3 = pd.read_csv('3.csv', header=None)
df = pd.concat([df0, df1, df2, df3])

# Setting the values x and y
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#training the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred = knn_classifier.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Printing the evaluation metrics
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)
