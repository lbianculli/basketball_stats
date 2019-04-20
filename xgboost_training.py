from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import random
import pickle
import warnings
from collections import defaultdict
import sklearn
import xgboost as xgb
from sklearn import linear_model, metrics, svm
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, validation_curve
from sklearn.externals import joblib
import os
import subprocess
import sys

BUCKET_NAME = 'xgboost-420'

#has to be google cloud dir
with open('/home/lbianculli123/xgboost_trainer/all_data-420.pkl', 'rb') as f:
    all_data = pickle.load(f)
    
train_data = all_data.sample(frac=.8)
test_data = all_data.drop(train_data.index)

try:
    train_labels = train_data[['home_win', 'margin', 'total_points']]
    train_data = train_data.drop(train_labels, axis=1)  # these are the three possible labels
    test_labels = test_data[['home_win', 'margin', 'total_points']]
    test_data = test_data.drop(test_labels, axis=1)
except Exception as e:
    pass

train_labels = np.asarray(train_labels['margin'], dtype=np.float64)  # or total points
test_labels = np.asarray(test_labels['margin'], dtype=np.float64)
normed_train_data = np.asarray((train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0), dtype=np.float64)
normed_test_data = np.asarray((test_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0), dtype=np.float64)

pca = PCA(n_components=30)
normed_train_data = pca.fit_transform(normed_train_data)
normed_test_data = pca.transform(normed_test_data)

# set up grid search
model = xgb.XGBRegressor()
scoring = 'neg_mean_squared_error'
n_estimators = [1000, 1250, 1500]
learning_rate = [0.05, 0.1, .2]
# gamma = [.5, 0.0, 5.0]
max_depth = [5, 7]
param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
kfold = StratifiedKFold(n_splits=5, shuffle=True)
grid_search = GridSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(normed_train_data, train_labels)
best_xgb = grid_result.best_estimator_

# Export the classifier to a file
model_filename = '/home/lbianculli/model.joblib'
joblib.dump(grid_result, model_filename)


gcs_model_path = os.path.join('gs://', BUCKET_NAME,
    datetime.datetime.now().strftime('iris_%Y%m%d_%H%M%S'), model_filename)
subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path],
    stderr=sys.stdout)

# summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# # run CV on the new model
# kfold_cv(normed_train_data, train_labels, best_xgb)
