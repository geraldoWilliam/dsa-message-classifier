import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import SelectFromModel
import pickle
from sys import argv
target = open("metadata", 'w')
#label

X = pd.read_csv('../feature_extraction/feature_balanced.csv')
y = pd.read_csv('../feature_extraction/balanced_label').values.ravel() #Force LOL

X_extra = pd.read_csv('../feature_extraction/feature_freq.csv')
y_extra = pd.read_csv('../feature_extraction/label').values.ravel() #Force LOL

# #remove extra feature -- ExtraTrees
#select=ExtraTreesClassifier()
#X = dataset['wkwk']
#X_new= select.fit(X,y)

# #remove extra feature -- SelectKBest
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
#
#select = SelectKBest(chi2, k=2)
#X_new = select.fit_transform(X, y) 

from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(X,y)
target.write("BNB Balanced Data score: ")
target.write(repr(bnb.score(X,y)))
target.write("\n")

predicted = cross_val_predict(bnb, X, y, cv=10)
target.write("BNB Balanced Data 10-Cross fold: ")
target.write(repr(metrics.accuracy_score(y, predicted)))
target.write("\n")

predicted = cross_val_predict(bnb, X_extra, y_extra, cv=10)
target.write("BNB Balanced Data 10-Cross fold All Data: ")
target.write(repr(metrics.accuracy_score(y_extra, predicted)))
target.write("\n")
pickle.dump(bnb, open("BNBAllFeature.p","wb"))

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(25, 100))
mlp.fit(X,y)
target.write("MLP Balanced Data score: ")
target.write(repr(mlp.score(X,y)))
target.write("\n")

predicted = cross_val_predict(mlp, X, y, cv=10)
target.write("MLP Balanced Data 10-Cross : ")
target.write(repr(metrics.accuracy_score(y, predicted)))
target.write("\n")

predicted = cross_val_predict(mlp, X_extra, y_extra, cv=10)
target.write("MLP Balanced Data 10-Cross Extra All Data: ")
target.write(repr(metrics.accuracy_score(y_extra, predicted)))
target.write("\n")
pickle.dump(mlp, open("MLPAllFeature.p","wb"))
target.close()
