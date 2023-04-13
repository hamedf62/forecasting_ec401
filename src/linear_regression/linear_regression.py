# import libraries
import numpy as np
from sklearn.linear_model import LinearRegression

# create some sample data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([3, 7, 11])

# create and fit the model
model = LinearRegression()
model.fit(X, y)

# make a prediction
X_new = np.array([[7, 8]])
y_new = model.predict(X_new)

print(y_new)
