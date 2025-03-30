import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import sklearn.model_selection
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures","absences"]]

predict = "G3"

X = np.array(data.drop([predict], axis = 1))
y = np.array(data[predict])
x_train, x_test, y_train , y_test = sklearn.model_selection.train_test_split(X,y, test_size = 0.1)

# #    Variable to store the highest accuracy found
# best = 0

# # Train the model 30 times to find the best-performing one
# for _ in range(30):
#     # Split the dataset into 90% training and 10% testing
#     x_train, x_test, y_train , y_test = sklearn.model_selection.train_test_split(X,y, test_size = 0.1)

#     # Initialize a linear regression model
#     linear = linear_model.LinearRegression()
    
#     #Train the model using regression model
#     linear.fit(x_train, y_train)
    
#     #Evaluate the model on the test set 
#     acc = linear.score(x_test, y_test)
#     print(acc) 
    
#     # If the current model has the highest accuracy so far, update the best model
#     if acc > best:
#         best = acc

#     # # Save the best-performing model to a file using pickle
#     with open("studentmodel.pickel", "wb") as f:
#         pickle.dump(linear, f)

pickle_in = open("studentmodel.pickel", "rb")
linear = pickle.load(pickle_in)

print("Cofficinet: \n ", linear.coef_)
print("Intercept: \n ", linear.intercept_)

prediction = linear.predict(x_test)

for x in range(len(prediction)):
    print(prediction[x], x_test[x], y_test[x])
    
p = "studytime" # Define the variable for the x-axis (study time)
style.use("ggplot") # Apply the 'ggplot' style for better visual aesthetics

# Create a scatter plot with 'studytime' on the x-axis and 'G3' (final grade) on the y-axis
plt.scatter(data[p], data["G3"])

plt.xlabel(p)#Label the x_axis
plt.ylabel("Final Grade") #Label the y_axis

plt.show()

