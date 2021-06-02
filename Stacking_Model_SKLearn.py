

#imports
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump, load
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
import numpy as np
import sys

#params
cupid_filename = "C:\\python\\cupid_users.csv"
anime_filename = "C:\\python\\anime_users.csv"
entity_filename = "C:\\python\\entity_users.csv"
twitter_filename = "C:\\python\\twitter_users.csv"
output_model_filename = "C:\\python\\stacking_model.joblib"
train_nrows = 50000

#functions
def get_stacking():
	# define the base models
  level0 = list()
  level0.append(('lr', LogisticRegression()))
  level0.append(('knn', KNeighborsClassifier()))
  level0.append(('cart', DecisionTreeClassifier()))
  level0.append(('svm', SVC()))
  #level0.append(('bayes', GaussianNB()))
  level0.append(('rfc',RandomForestClassifier()))
  # define meta learner model
  level1 = LogisticRegression()
  # define the stacking ensemble
  model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
  return model


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores

#main
def main():
  #for train:
  cupid = pd.read_csv(cupid_filename,nrows=train_nrows)
  anime = pd.read_csv(anime_filename,nrows=train_nrows)
  entity = pd.read_csv(entity_filename,nrows=train_nrows)
  #for test:
  twitter_test = pd.read_csv(twitter_filename)
  full_datasets = pd.concat([cupid,anime,entity])
  pd_ds = pd.DataFrame(full_datasets)
  print("vectorizing train data")
  number = preprocessing.LabelEncoder()
  y_tr = number.fit_transform(pd_ds['gender'])
  vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4))
  X_tr = vectorizer.fit_transform(pd_ds['name'])
  stacking_model = get_stacking()
  print("fitting stacking model")
  scores = stacking_model.fit(X_tr,y_tr)
  dump(stacking_model, output_model_filename)
  print("done training stacking model")
  #test on twitter data:
  print("vectorizing test data")
  y_test = number.fit_transform(twitter_test['gender'])
  vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4))
  X_test = vectorizer.fit_transform(twitter_test['name'])
  print("evaluating test data")
  scores_test = evaluate_model(stacking_model, X_test, y_test)
  print('done testing, test accuracy scoring vector:',scores_test)
  with open('output.txt', 'w') as f:
    print('test accurecy scoring vector:',scores_test, file=f)

if __name__ == "__main__":
    main()