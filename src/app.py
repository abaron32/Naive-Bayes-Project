### Naive Bayes Project

# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
import pickle


# Read csv
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews_dataset.csv')

## Transform dataframe

df_transf = df_raw.copy()

# Drop package_name column
df_transf = df_transf.drop('package_name', axis=1)

# Strip whitespaces from left and right sides from column review
df_transf['review'] = df_transf['review'].str.strip()

# column review to lower case
df_transf['review'] = df_transf['review'].str.lower()

df = df_transf.copy()

## Split dataframe

X = df['review']
y = df['polarity']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=25)

# use stratify because dataset is unbalanced


### Three options of pipelines depending on the preprocessing steps


# CountVectorizer
clf_1 = Pipeline([('cont_vect', CountVectorizer()), ('clf', MultinomialNB())])
clf_1.fit(X_train, y_train)
pred_1 = clf_1.predict(X_test)

# TfidfVectorizer
clf_2 = Pipeline([('tfidf_vect', TfidfVectorizer()), ('clf', MultinomialNB())])
clf_2.fit(X_train, y_train)
pred_2 = clf_2.predict(X_test)

# CountVectorizer and TfidfTransformer
clf_3 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
clf_3.fit(X_train, y_train)
pred_3 = clf_3.predict(X_test)


## Check results

print('CountVectorizer')
print(classification_report(y_test, pred_1))
print('TfidfVectorizer')
print(classification_report(y_test, pred_2))
print('CountVectorizer and TfidfTransformer')
print(classification_report(y_test, pred_3))

print('clf_1 Test Accuracy = ',metrics.accuracy_score(y_test,pred_1))
print('clf_2 Test Accuracy = ',metrics.accuracy_score(y_test,pred_2))
print('clf_3 Test Accuracy = ',metrics.accuracy_score(y_test,pred_3))

### Hyperparameter tuning

# CountVectorizer
n_iter_search = 5
parameters = {'cont_vect__ngram_range': [(1, 1), (1, 2)], 'clf__alpha': (1e-2, 1e-3)}
gs_clf_1 = RandomizedSearchCV(clf_1, parameters, n_iter = n_iter_search)
gs_clf_1.fit(X_train, y_train)
pred_1_grid = gs_clf_1.predict(X_test)


# TfidfVectorizer
n_iter_search = 5
parameters = {'clf__alpha': (1e-2, 1e-3)}
gs_clf_2 = RandomizedSearchCV(clf_2, parameters, n_iter = n_iter_search)
gs_clf_2.fit(X_train, y_train)
pred_2_grid = gs_clf_2.predict(X_test)


# CountVectorizer and TfidfTransformer
n_iter_search = 5
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}
gs_clf_3 = RandomizedSearchCV(clf_3, parameters, n_iter = n_iter_search)
gs_clf_3.fit(X_train, y_train)
pred_3_grid = gs_clf_3.predict(X_test)


## Check results
print('gs_clf_1')
print(classification_report(y_test, pred_1_grid))
print('gs_clf_2')
print(classification_report(y_test, pred_2_grid))
print('gs_clf_3')
print(classification_report(y_test, pred_3_grid))


## Best model
best_model = gs_clf_3.best_estimator_
print('The model with highest accuracy in the dataset, after hyperparameter tuning, is:', best_model)


## Save best model for future new data
pickle.dump(best_model, open('../models/best_model.pickle', 'wb'))