from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

import os
import joblib
import numpy as np
import pandas as pd

from ml_lib import SoftmaxRegression, MultinomialNaiveBayes, KNN, MulticlassSVM
from ensemble.decisiontree_id3 import DecisionTree
from ensemble.randomforest import RandomForest

MODEL_PATH = "model\level0"

df = pd.read_csv("data/preprocess/sentiment_train.csv")
df = df.dropna()

X_train = df['content'].values
y_train = df['label'].values

X_train = X_train.astype("str")
y_train = y_train.astype('int')

count_vector = CountVectorizer()
model_rf_preprocess = Pipeline([('vect', count_vector),
                    ('tfidf', TfidfTransformer()),
                    ])
X_train_CV = model_rf_preprocess.fit_transform(X_train)

df_test = pd.read_csv("data/preprocess/sentiment_test.csv")
df_test = df_test.dropna()
X_test = df_test['content'].values
y_test = df_test['label'].values

X_test = X_test.astype("str")
y_test = y_test.astype('int')

X_test_CV = model_rf_preprocess.transform(X_test)

smoteTomek = SMOTETomek()
X_train_CV_smote, y_train_smote = smoteTomek.fit_resample(X_train_CV, y_train)
print('Resampled dataset shape %s' % Counter(y_train_smote))

rus = RandomUnderSampler()
X_test_CV, y_test = rus.fit_resample(X_test_CV, y_test)
print('Resampled dataset shape %s' % Counter(y_test))

def train(model, name, smote=False):
    if smote:
        model.fit(X_train_CV_smote.toarray(), y_train_smote)
    else:
        model.fit(X_train_CV.toarray(), y_train)
    predict = model.predict(X_test_CV.toarray())
    print(type(model).__name__)
    joblib.dump(model, os.path.join(MODEL_PATH, name))   
    print(classification_report(y_test, predict))
    

# KNN - SMOTETomek - Too large(1.8GB)
# knn = KNeighborsClassifier(n_neighbors=7, weights='uniform')
# train(knn, "knn.sav", smote=True)

# Softmax - Normal
sr = SoftmaxRegression(lr=0.3, max_epochs=10000)
train(sr, "softmax.sav")

# SVM - SMOTETomek
svm = MulticlassSVM(reg=0.1, lr=0.2, beta1=0.9, beta2=0.999, eps=1e-6, max_epochs=300)
train(svm, "svm.sav", smote=True)

# Decision Tree - Normal
dt = DecisionTreeClassifier()
train(dt, "tree.sav")

# Random Forest - SMOTETomek
rf = RandomForestClassifier(n_estimators=128, random_state=42, max_depth=15)
train(rf, "forest.sav", smote=True)

# MultinomialNB - SMOTETomek
nb = MultinomialNaiveBayes(alpha=2.0)
train(nb, "nb.sav", smote=True)



