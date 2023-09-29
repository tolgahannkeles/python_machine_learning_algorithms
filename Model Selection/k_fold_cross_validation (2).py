"""
K-fold cross validation
https://prnt.sc/QxKP7JE-fCto
"""


import pandas as pd

data=pd.read_csv("C:/Stupid Humanity/Steps of AI Advanture/MachineLearning/Classification/decisionTree/veriler.csv")


x=data[["boy","kilo","yas"]]
y=data[["cinsiyet"]]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred_lr=lr.predict(x_test)


# k_fold_cross_validation

"""
estimator: algoritma objesi
X:
y:
cv: kaç parçaya bölmeli
"""

from sklearn.model_selection import cross_val_score
c_vs=cross_val_score(estimator=lr,X=x_train,y=y_train,cv=3) 
print(c_vs.mean()) # accuracy değeri verir