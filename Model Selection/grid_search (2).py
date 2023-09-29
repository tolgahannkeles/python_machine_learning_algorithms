
import pandas as pd

data=pd.read_csv("C:/Software&Design/MachineLearning/News/Classification/decisionTree/veriler.csv")


x=data[["boy","kilo","yas"]]
y=data[["cinsiyet"]]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)


from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
# k_fold_cross_validation

from sklearn.model_selection import cross_val_score
c_vs=cross_val_score(estimator=svc,X=x_train,y=y_train,cv=3) 
print(c_vs.mean()) # accuracy değeri verir

from sklearn.model_selection import GridSearchCV

parameters=[{"C":[1,2,3,4,5],"kernel":["linear"]},
           {"C":[1,10,100,1000],"kernel":["rbf"],"gamma":[1,0.5,0.1,0.01,0.001]}]


"""
GSCV Parameters
estimator: algoritma objesi
param_grid: arametre listesi
scoring: scorlama fonksiyonu çrn: accuracy
cv: listenin kaça bölüneceği
n_jobs: aynı anda yapılacak iş
"""
gs=GridSearchCV(estimator=svc,
                param_grid=parameters,
                scoring="accuracy",
                cv=10,
                n_jobs=-1)

gs.fit(x_train,y_train)
bestResult=gs.best_score_
bestParams=gs.best_params_

print(bestResult)
print(bestParams)