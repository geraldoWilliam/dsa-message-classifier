import pandas as pd
import numpy as np

#from sklearn.feature_selection import SelectFromModel
#label
X = pd.read_csv('../feature_extraction/feature_balanced.csv')
y = pd.read_csv('../feature_extraction/balanced_label.csv')

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
bnb.score(X,y)
predicted = cross_val_predict(bnb, X, y, cv=10)
print(metrics.accuracy_score(y, predicted))
pickle.dump(mlp, open("BNBAllFeature.p","wb"))

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(25, 100))
mlp.fit(X,y)
mlp.score(X,y)
predicted = cross_val_predict(mlp, X, y, cv=10)
print(metrics.accuracy_score(y, predicted))
pickle.dump(mlp, open("MLPAllFeature.p","wb"))
