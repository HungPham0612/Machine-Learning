import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import sklearn.model_selection
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures","absences"]]

predict = "G3"

X = np.array(data.drop([predict], axis = 1))
y = np.array(data[predict])

x_train, x_test, y_train , y_test = sklearn.model_selection.train_test_split(X,y, test_size = 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)

acc = linear.score(x_test, y_test)

print(acc)

print("Cofficinet: \n ", linear.coef_)
print("Intercept: \n ", linear.intercept_)

prediction = linear.predict(x_test)

for x in range(len(prediction)):
    print(prediction[x], x_test[x], y_test[x])