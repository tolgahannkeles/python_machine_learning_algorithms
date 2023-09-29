import pandas as pd
from sklearn.metrics import confusion_matrix

data=pd.read_excel("C:/Software&Design/MachineLearning/News/Classification/assignment/Iris.xls")

"""
lr, k-nn, svc, naive bayes, dt, rf, 

en verimliyi bul
"""

x=data.iloc[:,:4]
y=data.iloc[:,-1]


from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

from sklearn.linear_model import LogisticRegression # [[16  0  0], [ 0 18  1], [ 0  0 15]] #1
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

from sklearn.neighbors import KNeighborsClassifier # [[16  0  0] ,[ 0 18  1], [ 0  0 15]] #1
knn=KNeighborsClassifier(n_neighbors=5,metric="minkowski")
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

from sklearn.svm import SVC #[[16  0  0] ,[ 0 18  1] ,[ 0  1 14]] #2
svc=SVC(kernel="rbf")
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)

from sklearn.naive_bayes import MultinomialNB # [[16  0  0] ,[ 0  4 15] ,[ 0  0 15]]  #4
mnb=MultinomialNB()
mnb.fit(x_train,y_train)
y_pred=mnb.predict(x_test)

from sklearn.naive_bayes import GaussianNB #[[16  0  0] ,[ 0 19  0] ,[ 0  2 13]] #2
gnb=GaussianNB()
gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_test)

from sklearn.naive_bayes import BernoulliNB # [[ 0  0 16] ,[ 0  0 19] ,[ 0  0 15]] #baya
bnb=BernoulliNB()
bnb.fit(x_train,y_train)
y_pred=bnb.predict(x_test)

from sklearn.tree import DecisionTreeClassifier #[[16  0  0] ,[ 0 18  1] ,[ 0  1 14]] #2
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn.tree import DecisionTreeClassifier #[[16  0  0] ,[ 0 18  1] ,[ 0  1 14]] #2
dt=DecisionTreeClassifier(criterion="gini")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn.ensemble import RandomForestClassifier #[[16  0  0] ,[ 0 18  1] ,[ 0  1 14]] #2
rfc=RandomForestClassifier(criterion="gini",n_estimators=4)
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)


cm=confusion_matrix(y_pred=y_pred,y_true=y_test)

print(cm)

