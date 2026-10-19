import csv
import random
import pandas as pd
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from ucimlrepo import fetch_ucirepo 


#model = Perceptron()
# model = svm.SVC()
# model = KNeighborsClassifier(n_neighbors=1)
model = GaussianNB()

# fetch dataset 
secondary_mushroom = fetch_ucirepo(id=848) 
  
# data (as pandas dataframes) 
X = secondary_mushroom.data.features 
y = secondary_mushroom.data.targets 
X = pd.get_dummies(X)

X_training, X_testing, y_training, y_testing = train_test_split(
    X, y, test_size=0.10, random_state=42
)

y_training = y_training.values.ravel()
y_testing = y_testing.values.ravel()

scaler = StandardScaler()
X_training = scaler.fit_transform(X_training)
X_testing = scaler.fit_transform(X_testing)

# Fit model
model.fit(X_training, y_training)

# Make predictions on the testing set
predictions = model.predict(X_testing)

# Compute how well we performed
correct = (y_testing == predictions).sum()
incorrect = (y_testing != predictions).sum()
total = len(predictions)

# Print results
print(f"Results for model {type(model).__name__}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {100 * correct / total:.2f}%")
