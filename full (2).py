import pandas as pd
import numpy as np

path_log="C:/Users/tolga/Desktop/results.txt"
path_grid="C:/Users/tolga/Desktop/grid_result.txt"
# Preprocessing
data="data"

from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan, strategy="mean")
data[["a"]]=imputer.fit_transform(data[["a"]])


x="x"
y="y"

print(data.corr())
import statsmodels.api as sm
model=sm.OLS(y,x).fit()
print(model.summary())


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)


from sklearn.model_selection import GridSearchCV

############################### REGRESSION ###############################
from sklearn.metrics import r2_score
#--------------- Linear Regression ---------------

from sklearn.linear_model import LinearRegression
lr=LinearRegression(positive=False) 
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

with open(path_log,"a") as file:
    file.write(f"Linear Regression: {r2_score(y_true=y_test,y_pred=y_pred)}\n")

print("Linear Regression is finished.")

#--------------- Polynomial Regression ---------------

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_regr=PolynomialFeatures(degree=31)
x_poly=poly_regr.fit_transform(x)

lr=LinearRegression(positive=False)
lr.fit(x_poly,y)
y_pred=lr.predict(x_test)

with open(path_log,"a") as file:
    file.write(f"Polynomial Regression: {r2_score(y_true=y_test,y_pred=y_pred)}\n")

print("Polynomial Regression is finished.")


#--------------- Support Vector Regression ---------------

from sklearn.svm import SVR
svr=SVR()

params=[{"kernel":["rbf","linear","poly","sigmoid"], "C":[np.arange(0.1,10,0.1)]}]

gs=GridSearchCV(estimator=svr,
                param_grid=params,
                scoring="accuracy",
                cv=10,
                n_jobs=-1)

gs.fit(x_train,y_train)
bestResult=gs.best_score_
bestParams=gs.best_params_
with open(path_grid,"a") as file:
    file.write(f"Support Vector Regression Best Result: {bestResult}\n")
    file.write(f"Support Vector Regression Best Params: {bestParams}\n")

svr.fit(X_train,y_train)
y_pred=svr.predict(X_test)

with open(path_log,"a") as file:
    file.write(f"Support Vector Regression: {r2_score(y_true=y_test,y_pred=y_pred)}\n")

print("Support Vector Regression is finished.")


#--------------- Decision Tree Regression ---------------

from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(random_state=0)

params=[{"criterion":["squared_error", "friedman_mse", "absolute_error", "poisson"], "splitter":["best","random"]}]

gs=GridSearchCV(estimator=dtr,
                param_grid=params,
                scoring="accuracy",
                cv=10,
                n_jobs=-1)

gs.fit(x_train,y_train)
bestResult=gs.best_score_
bestParams=gs.best_params_
with open(path_grid,"a") as file:
    file.write(f"DecisionTreeRegressor Best Result: {bestResult}\n")
    file.write(f"DecisionTreeRegressor Best Params: {bestParams}\n")


dtr.fit(x_train,y_train)
y_pred=dtr.predict(x_test)

with open(path_log,"a") as file:
    file.write(f"Decision Tree Regression: {r2_score(y_true=y_test,y_pred=y_pred)}\n")

print("Decision Tree Regression is finished.")


#--------------- Random Forest Regression ---------------

from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=20,random_state=0) # decision tree sayısını gir

params=[{"n_estimators":[1,5,10,20,50,100], "criterion":["squared_error", "friedman_mse", "absolute_error", "poisson"]}]

gs=GridSearchCV(estimator=rfr,
                param_grid=params,
                scoring="accuracy",
                cv=10,
                n_jobs=-1)

gs.fit(x_train,y_train)
bestResult=gs.best_score_
bestParams=gs.best_params_
with open(path_grid,"a") as file:
    file.write(f"RandomForestRegressor Best Result: {bestResult}\n")
    file.write(f"RandomForestRegressor Best Params: {bestParams}\n")


rfr.fit(x,y)
y_pred=rfr.predict(x_test)

with open(path_log,"a") as file:
    file.write(f"Random Forest Regression: {r2_score(y_true=y_test,y_pred=y_pred)}\n")

print("Random Forest Regression is finished.")



############################### CLASSIFICATION ###############################
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix as cm
#--------------- Logistic Regression ---------------

from sklearn.linear_model import LogisticRegression
logr= LogisticRegression()
logr.fit(x_train,y_train)
y_pred=logr.predict(x_test)

with open(path_log,"a") as file:
    file.write(f"Logistic Regression: \n{cm(y_true=y_test,y_pred=y_pred)}\n")

print("LogisticRegression is finished.")


#--------------- K-NN ---------------

from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier(n_neighbors=5, metric="minkowski") # n_neighbors=l(en(x_train)**0.5)/2


params=[{"n_neighbors":[1,5,10,50,100], "weights":["uniform", "distance"],"algorithm":["auto","ball_tree","kd_tree", "brute"],"metric":["minkowski","euclidean","manhattan","chebyshev","wminkowski","mahalanobis","seuclidean"]}]


gs=GridSearchCV(estimator=knc,
                param_grid=params,
                scoring="accuracy",
                cv=10,
                n_jobs=-1)

gs.fit(x_train,y_train)
bestResult=gs.best_score_
bestParams=gs.best_params_
with open(path_grid,"a") as file:
    file.write(f"KNeighborsClassifier Best Result: {bestResult}\n")
    file.write(f"KNeighborsClassifier Best Params: {bestParams}\n")

knc.fit(x_train,y_train)
y_pred=knc.predict(x_test)

with open(path_log,"a") as file:
    file.write(f"KNeighborsClassifier: \n{cm(y_true=y_test,y_pred=y_pred)}\n")

print("KNeighborsClassifier is finished.")

#--------------- Naive Bayes ---------------
"""
GaussianNB
    -continuos değerlerde kullanılır
MultinomialNB
    - multinominal değerlerde kullanılır
BernoulliNB
    -Bynominal datalarda kullanılır
"""

from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()
gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_test)
with open(path_log,"a") as file:
    file.write(f"GaussianNB : \n{cm(y_true=y_test,y_pred=y_pred)}\n")

from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
mnb.fit(x_train,y_train)
y_pred=mnb.predict(x_test)

with open(path_log,"a") as file:
    file.write(f"MultinomialNB : \n{cm(y_true=y_test,y_pred=y_pred)}\n")

from sklearn.naive_bayes import BernoulliNB
bnb=BernoulliNB()
bnb.fit(x_train,y_train)
y_pred=bnb.predict(x_test)

with open(path_log,"a") as file:
    file.write(f"BernoulliNB : \n{cm(y_true=y_test,y_pred=y_pred)}\n")

print("Naive Bayes is finished.")



#--------------- SUPPORT VECTOR CLASSIFIER ---------------

from sklearn.svm import SVC
svc=SVC(kernel="rbf")

params=[{"C":[np.arange(0.1,10,0.1)], "kernel":["linear", "poly", "rbf", "sigmoid"]}]


gs=GridSearchCV(estimator=svc,
                param_grid=params,
                scoring="accuracy",
                cv=10,
                n_jobs=-1)

gs.fit(X_train,y_train)
bestResult=gs.best_score_
bestParams=gs.best_params_
with open(path_grid,"a") as file:
    file.write(f"SVC Best Result: {bestResult}\n")
    file.write(f"SVC Best Params: {bestParams}\n")

svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)

with open(path_log,"a") as file:
    file.write(f"SVC : \n{cm(y_true=y_test,y_pred=y_pred)}\n")

print("SVC is finished.")

#--------------- DECISION TREE CLASSIFIER ---------------

from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(criterion="entropy") # defaultu gini

params=[{"criterion":["gini", "entropy", "log_loss"], "splitter":["best","random"]}]

gs=GridSearchCV(estimator=dtc,
                param_grid=params,
                scoring="accuracy",
                cv=10,
                n_jobs=-1)

gs.fit(x_train,y_train)
bestResult=gs.best_score_
bestParams=gs.best_params_
with open(path_grid,"a") as file:
    file.write(f"DecisionTreeClassifier Best Result: {bestResult}\n")
    file.write(f"DecisionTreeClassifier Best Params: {bestParams}\n")

dtc.fit(x_train,y_train)
y_pred=dtc.predict(x_test)

with open(path_log,"a") as file:
    file.write(f"DecisionTreeClassifier : \n{cm(y_true=y_test,y_pred=y_pred)}\n")

print("DecisionTreeClassifier is finished.")


#--------------- RANDOM FOREST CLASSIFIER ---------------

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(criterion="entropy",n_estimators=10)

params=[{"n_estimators":[1,5,10,50,100], "criterion":["gini", "entropy", "log_loss"]}]

gs=GridSearchCV(estimator=rfc,
                param_grid=params,
                scoring="accuracy",
                cv=10,
                n_jobs=-1)

gs.fit(x_train,y_train)
bestResult=gs.best_score_
bestParams=gs.best_params_
with open(path_grid,"a") as file:
    file.write(f"RandomForestClassifier Best Result: {bestResult}\n")
    file.write(f"RandomForestClassifier Best Params: {bestParams}\n")



rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)

with open(path_log,"a") as file:
    file.write(f"RandomForestClassifier : \n{cm(y_true=y_test,y_pred=y_pred)}\n")

print("RandomForestClassifier is finished.")


#--------------- XGBOOST ---------------

from xgboost import XGBClassifier
xcb=XGBClassifier()
xcb.fit(x_train,y_train)
y_pred=xcb.predict(x_test)

with open(path_log,"a") as file:
    file.write(f"XGBClassifier : \n{cm(y_true=y_test,y_pred=y_pred)}\n")

print("XGBClassifier is finished.")

############################### ARTIFICAL NEURAL NETWORKS ###############################

#--------------- Classifier ---------------

from keras.models import Sequential
from keras.layers import Dense

#hidden layerlarda linear functionlar kullan output layerlarda sigmoid functionlar kullan

classifier=Sequential()
classifier.add(Dense(6,activation="relu",input_dim=12)) #1.hidden layer 
classifier.add(Dense(6,activation="relu")) #2.hidden layer 
classifier.add(Dense(1,activation="sigmoid")) #output layer

classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

classifier.fit(X_train,y_train,epochs=50)
y_pred=classifier.predict(X_test)

with open(path_log,"a") as file:
    file.write(f"ANN Classifier : \n{cm(y_true=y_test,y_pred=y_pred)}\n")
