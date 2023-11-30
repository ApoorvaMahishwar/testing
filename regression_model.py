from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import pickle
import os

X, y = make_regression(10000,n_features = 10)

# Train a model
regession = LinearRegression().fit(X, y.ravel())
# Print out training r2
print(regession.score(X,y.ravel() ))

# Write the model to a file
if not os.path.isdir("models/"):
    os.mkdir("models")

filename = 'models/model.pkl'
pickle.dump(regession, open(filename, 'wb'))