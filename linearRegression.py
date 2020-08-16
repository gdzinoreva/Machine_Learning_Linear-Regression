#This code uses the sklearn diabetes dataset to perform linear regression
#to find the best fit line through the data.
#The linear regression model is performed using a function that utilizes y=mx+b


#importing necessary modules
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn import linear_model


#defining function to get coefficients of the line equation
def constants(x, y):
    grad = (np.mean(x)*np.mean(y)-np.mean(x*y))/((np.mean(x))**2-np.mean(x**2))
    intercept = np.mean(y) - grad*np.mean(x)
    return (grad, intercept)


#creating figure with two plots    
fig = plt.figure()
ax1 = plt.subplot(2,1,1)
ax1.set(title = "Training Data Plot",
        ylabel = "Disease Progression")

ax2 = plt.subplot(2,1,2)
ax2.set(title = "Testing Data Plot",
        xlabel = "variable",
        ylabel = "Disease Progression")


#loading dataset and selecting data to use
d = load_diabetes()
d_X = d.data[:, np.newaxis, 2]
dx_train = d_X[:-20]
dy_train = d.target[:-20].reshape(-1,1)
dx_test = d_X[-20:]
dy_test = d.target[-20:].reshape(-1,1)


#calling function to calculate constants of the data being used
m,b = constants(dx_train, dy_train)
print("Coefficients:\n", m, "\n", b)


#apply line of best fit equation
y_train = m * dx_train + b
y_test = m * dx_test + b


#mean square error
mse = np.mean((y_test - dy_test) **2)
print("MSE: ", mse)


#plotting
ax1.scatter(dx_train, dy_train, c = "r")
ax1.plot(dx_train, y_train, c = "b")
ax1.legend(["line of best fit"])
ax2.scatter(dx_test, dy_test, c = "g")
ax2.plot(dx_test, y_test,  c  = "b")
ax2.legend(["line of best fit"])
plt.show()
