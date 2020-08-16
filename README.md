# Machine_Learning_Linear-Regression

Regression analysis is a statistical process used to estimate the relationship between variables. The point of **Linear regression** is to find ‘the line of best fit’, that is to say,if we have scattered points plotted on a graph, we want to draw a line through them so that as many of the dots as possible are as close to the line as possible.

This code uses the **Diabetes Dataset** that comes with sklearn which consists of 10 physiological variables (age, sex, weight and blood pressure) which were measured on 442 patients, and are an indication of the disease progression after one year. The goal is to predict disease progression from physiological variables.

Instead of using *linear_model.LinearRegression() function* from sklearn I have written a function that makes use of numpy to calculate the gradient and they-intercept of the best
fit line, which has equation y = mx + b. The equations below describe how both the gradient and the y-intercept can be calculated from the training data and labels.

  ○ m = (μ(x) * μ(y) − μ(x * y))/((μ(x))2 − μ(x2))
  ○ b = μ(y) − m * μ(x)
Where μ is a mean function
