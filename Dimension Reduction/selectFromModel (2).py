from statistics import mode
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
import pandas as pd

data=pd.read_csv("C:/Stupid Humanity/Steps of AI Advanture/MachineLearning/Dimension Reduction/wine.csv")

x=data.iloc[:,:-1]
y=data.iloc[:,-1]

lsvc=LinearSVC(C=0.01, penalty="l1", dual=False).fit(x, y)
model= SelectFromModel(lsvc, prefit=True)
x_model= model.transform(x)



# ağaç temelli

from sklearn.ensemble import ExtraTreesClassifier

clf= ExtraTreesClassifier(n_estimators=50).fit(x,y)
clf.feature_importances_
model=SelectFromModel(clf, prefit=True)
x_model=model.transform(x)



print(x.shape)
print(x_model.shape)
