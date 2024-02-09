import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,mean_squared_error,r2_score

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

#training for decsion tree classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train,y_train)

#training for random forest
rf_classifier = RandomForestClassifier(n_estimators=20, random_state=0, criterion='entropy')
rf_classifier.fit(X_train, y_train)

#training for svc
svm_classifier=SVC()
svm_classifier.fit(X_train, y_train)



# Predicting the test set results
y_pred_dt =dt_classifier.predict(X_test)
y_pred_knn = knn_classifier.predict(X_test)
y_pred_rf = rf_classifier.predict(X_test)
y_pred_svm = svm_classifier.predict(X_test)


# Compare the performance of both models using cross-validation
accuracies_dt = cross_val_score(estimator=dt_classifier, X=X, y=Y, cv=10)
accuracies_knn = cross_val_score(estimator=knn_classifier, X=X, y=Y, cv=10)
accuracies_rf = cross_val_score(estimator=rf_classifier, X=X, y=Y, cv=10)
accuracies_svm = cross_val_score(estimator=svm_classifier, X=X, y=Y, cv=10)

# Printing the accuracies
print("Decision Tree Classifier:")
print("Mean accuracy:", accuracies_dt.mean())
print("Standard deviation of accuracies:", accuracies_dt.std())

print("\nK-Nearest Neighbors (KNN) Classifier:")
print("Mean accuracy:", accuracies_knn.mean())
print("Standard deviation of accuracies:", accuracies_knn.std())

print("\nRandom Forest Classifier:")
print("Mean accuracy:", accuracies_rf.mean())
print("Standard deviation of accuracies:", accuracies_rf.std())

print("\nStandard Vector Machine(SVM):")
print("Mean accuracy:", accuracies_svm.mean())
print("Standard deviation of accuracies:", accuracies_svm.std())

# Plotting the comparison of accuracies
plt.figure(figsize=(8, 6))
plt.bar(['Decision Tree', 'KNN','Random Forest','SVM'], [accuracies_dt.mean(), accuracies_knn.mean(),accuracies_rf.mean(),accuracies_svm.mean()], yerr=[accuracies_dt.std(), accuracies_knn.std(),accuracies_rf.std(),accuracies_svm.std()])
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Comparison of Mean Accuracy between Decision Tree , KNN , Random forest and SVM')
plt.show()
