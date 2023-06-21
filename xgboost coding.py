# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 21:19:32 2023

@author: Dell
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"D:\Data Science\Daily Practice\April\11-04-2023\7.XGBOOST\Churn_Modelling.csv")

dataset
x=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values

print(x)
print(y)

dataset.isnull().sum().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])

print(x)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

print(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from xgboost import XGBClassifier

classifier=XGBClassifier()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac= accuracy_score(y_test, y_pred) 
print(ac)

bias=classifier.score(x_train,y_train)
print(bias)

variance=classifier.score(x_test,y_test)
print(variance)

#

classifier1=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.00000001, max_delta_step=0, max_depth=10,
              min_child_weight=1,monotone_constraints='()',
              n_estimators=150, n_jobs=0, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)

classifier1.fit(x_train,y_train)
y_pred1 = classifier1.predict(x_test)

from sklearn.metrics import confusion_matrix
cm1= confusion_matrix(y_test,y_pred1)
print(cm1)

from sklearn.metrics import accuracy_score
ac1= accuracy_score(y_test, y_pred1) 
print(ac1)

bias1=classifier1.score(x_train,y_train)
print(bias1)

variance1=classifier1.score(x_test,y_test)
print(variance1)


classifier2=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.00000001, max_delta_step=0, max_depth=10,
              min_child_weight=1,monotone_constraints='()',
              n_estimators=250, n_jobs=0, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)

classifier2.fit(x_train,y_train)
y_pred2 = classifier2.predict(x_test)

from sklearn.metrics import confusion_matrix
cm2= confusion_matrix(y_test,y_pred2)
print(cm2)

from sklearn.metrics import accuracy_score
ac2= accuracy_score(y_test, y_pred2) 
print(ac2)

bias2=classifier2.score(x_train,y_train)
print(bias2)

variance2=classifier2.score(x_test,y_test)
print(variance2)



classifier3=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.001, max_delta_step=0, max_depth=18,
              min_child_weight=1,monotone_constraints='()',
              n_estimators=500, n_jobs=0, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)

classifier3.fit(x_train,y_train)
y_pred3 = classifier3.predict(x_test)

from sklearn.metrics import confusion_matrix
cm3= confusion_matrix(y_test,y_pred3)
print(cm3)

from sklearn.metrics import accuracy_score
ac3= accuracy_score(y_test, y_pred3) 
print(ac3)

bias3=classifier3.score(x_train,y_train)
print(bias3)

variance3=classifier3.score(x_test,y_test)
print(variance3)



classifier4=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.0000001, max_delta_step=0, max_depth=10,
              min_child_weight=1,monotone_constraints='()',
              n_estimators=700, n_jobs=0, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)

classifier4.fit(x_train,y_train)
y_pred4 = classifier4.predict(x_test)

from sklearn.metrics import confusion_matrix
cm4= confusion_matrix(y_test,y_pred4)
print(cm4)

from sklearn.metrics import accuracy_score
ac4= accuracy_score(y_test, y_pred4) 
print(ac4)

bias4=classifier4.score(x_train,y_train)
print(bias4)

variance4=classifier4.score(x_test,y_test)
print(variance4)


classifier5=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.0000001, max_delta_step=0, max_depth=10,
              min_child_weight=1,monotone_constraints='()',
              n_estimators=225, n_jobs=0, num_parallel_tree=1, random_state=41,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)

classifier5.fit(x_train,y_train)
y_pred5 = classifier5.predict(x_test)

from sklearn.metrics import confusion_matrix
cm5= confusion_matrix(y_test,y_pred5)
print(cm5)

from sklearn.metrics import accuracy_score
ac5= accuracy_score(y_test, y_pred5) 
print(ac5)

bias5=classifier5.score(x_train,y_train)
print(bias5)

variance5=classifier5.score(x_test,y_test)
print(variance5)