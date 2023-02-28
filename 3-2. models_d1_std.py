import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,recall_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from collections import defaultdict
import pickle

df = pd.read_csv('enc_stand.csv')
y = df.사건
x = df.drop('사건',axis=1)

# 모델1. xgboost
# 모델 학습 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y)

# Define the model
xgb_model = XGBClassifier()

# Define the parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.1, 0.2, 0.3],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.5, 0.8, 1.0],
    'colsample_bytree': [0.5, 0.8, 1.0]
}

# Define the GridSearchCV object
grid_search = GridSearchCV(xgb_model, param_grid, scoring='recall', cv=5, n_jobs=-1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Predict the target values for the test data using the best model
y_pred = grid_search.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test,y_pred)

print("Best parameters of xgboost: {}".format(grid_search.best_params_))
print("Best cross-validation score of xgboost: {:.2f}".format(grid_search.best_score_))
print("XGboost Accuracy: {:.2f}%".format(accuracy * 100))
print("XGboost Recall: {:.2f}%".format(recall*100))


# 모델 저장
best_xgb = grid_search.best_estimator_
filename = 'best_xgb_d1.pkl'
with open(filename, 'wb') as file:
    pickle.dump(best_xgb, file)


# 모델 2. random forest
# Define the model
clf = RandomForestClassifier()

# specify the hyperparameter grid to search over
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}

# create a GridSearchCV object
grid_search = GridSearchCV(clf, param_grid, cv=5,scoring='recall')

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Predict the target values for the test data using the best model
y_pred = grid_search.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test,y_pred)

print("Best parameters of RF: {}".format(grid_search.best_params_))
print("Best cross-validation score of RF: {:.2f}".format(grid_search.best_score_))
print("RandomForest Accuracy: {:.2f}%".format(accuracy * 100))
print("RandomForest Recall: {:.2f}%".format(recall*100))


# 모델 저장
best_rf = grid_search.best_estimator_
filename = 'best_rf_d1.pkl'
with open(filename, 'wb') as file:
    pickle.dump(best_rf, file)


#model3. SVM
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': [0.1, 1, 10]}

# Create an SVM model
svm = SVC()

# Create a GridSearchCV object
grid_search = GridSearchCV(svm, param_grid, cv=5)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test,y_pred)

# Print the best parameters and the corresponding mean cross-validated score
print("Best parameters of svm: {}".format(grid_search.best_params_))
print("Best cross-validation score of svm: {:.2f}".format(grid_search.best_score_))
print("SVM Accuracy: {:.2f}%".format(accuracy * 100))
print("SVM Recall: {:.2f}%".format(recall*100))


# 모델 저장
best_svm = grid_search.best_estimator_
filename = 'best_svm_d1.pkl'
with open(filename, 'wb') as file:
    pickle.dump(best_svm, file)



# 모델 4. naive bayes 
# Define the model
model = GaussianNB()

# Define the hyperparameters to search
param_grid = {'var_smoothing': np.logspace(0, -9, num=100)}

# Create the grid search object
grid_search = GridSearchCV(model, param_grid, cv=5,scoring = 'recall')

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Make predictions on the test data
y_pred = grid_search.predict(X_test)

# Print the accuracy
acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test,y_pred)

print("Best parameters of NB: {}".format(grid_search.best_params_))
# print("Best cross-validation score of NB: {:.2f}".format(grid_search.best_score_))
print("Naive bayes Accuracy: {:.2f}%".format(acc * 100))
print("Naive bayes Recall: {:.2f}%".format(recall*100))

# 모델 저장
best_nb = grid_search.best_estimator_
filename = 'best_nb_d1.pkl'
with open(filename, 'wb') as file:
    pickle.dump(best_nb, file)