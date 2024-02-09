import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,classification_report,accuracy_score,confusion_matrix

# Reading the data and combining them
df0 = pd.read_csv('0.csv', header=None)
df1 = pd.read_csv('1.csv', header=None)
df2 = pd.read_csv('2.csv', header=None)
df3 = pd.read_csv('3.csv', header=None)
df = pd.concat([df0, df1, df2, df3])

# Setting the values x and y
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

# Splitting and scaling the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#evaluation metrics
print('Mean Squared Error:', mse)
print('R2 Score:', r2)

# cross val score , accuracy and classification report
X_train = df.values
res = cross_val_score(estimator=regressor, X=X_train, y=Y, cv=10)
print("Cross Validation Scores:", res)



# Plotting the regression results
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted')
plt.grid(True)
plt.show()
