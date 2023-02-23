import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,recall_score
from xgboost import XGBClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
from collections import defaultdict
import pickle

df = pd.read_csv('enc_not_stand_ch.csv')
y = df.사건
x = df.drop('사건',axis=1)

# 모델1. xgboost
# 모델 학습 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y)

# Define the model
xgb_model = XGBClassifier()

# Define the parameter grid
param_grid = {
    'max_depth': [5, 7, 9],
    'n_estimators': [300,500,1000],
    'learning_rate': [0.3,0.5,0.7],
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
filename = 'best_xgb_d2ch_fi.pkl'
with open(filename, 'wb') as file:
    pickle.dump(best_xgb, file)


importance =grid_search.get_booster().get_score(importance_type='gain')

# Print the feature importance scores in descending order
for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f'{feature}: {score}')