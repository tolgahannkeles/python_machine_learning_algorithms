"""
(10 yıl tecrübe, 100 puan, CEO) [[10,10,100]]
(10 yıl tecrübe, 100 puan, Müdür) [[7,10,100]]

MLR, PR, SVR, DT, RF
"""

ceo=[[10,10,100]]
mudur=[[7,10,100]]

import pandas as pd
from sklearn.metrics import r2_score

data=pd.read_csv("C:/Software&Design/MachineLearning/News/predictionFinal/salary.csv")

# data=data[["UnvanSeviyesi","Kidem","Puan","maas"]]

x=data[["UnvanSeviyesi","Kidem","Puan"]]
y=data[["maas"]]


"""
# p-value bakma
import statsmodels.api as sm

model=sm.OLS(y,x).fit()
print(model.summary())
"""

# train test ayırma
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test= train_test_split(x,y,test_size=0.33,random_state=15)
#----------------------------------------------------------
"""
# MLR
from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x_train,y_train)


r2_score(y_test,lr.predict(x_test)) #0.5244511103680498
print(lr.predict(ceo))
print(lr.predict(mudur))



#----------------------------------------------------------

# PR
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
x_polyTrain=poly.fit_transform(x_train)
x_polyTest=poly.fit_transform(x_test)

from sklearn.linear_model import LinearRegression
lr_poly=LinearRegression()
lr_poly.fit(x_polyTrain,y_train)

r2_score(y_test,lr_poly.predict(x_polyTest))  # 0.32016154806067343
poly_ceo=poly.fit_transform(ceo)
poly_mudur=poly.fit_transform(mudur)
print(lr_poly.predict(poly_ceo))
print(lr_poly.predict(poly_mudur))

#----------------------------------------------------------

# SVR
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(x_train)
X_test=scaler.fit_transform(x_test)
Y_train=scaler.fit_transform(y_train)
Y_test=scaler.fit_transform(y_test)


from sklearn.svm import SVR
svr=SVR(kernel="rbf")
svr.fit(X_train,Y_train)

r2_score(Y_test,svr.predict(X_test)) # 0.4813739587486682

scaled_ceo=scaler.fit_transform(ceo)
scaled_mudur=scaler.fit_transform(mudur)
print(svr.predict(scaled_ceo))
print(svr.predict(scaled_mudur))

#----------------------------------------------------------

#DT

from sklearn.tree import DecisionTreeRegressor

r_dt=DecisionTreeRegressor()
r_dt.fit(x_train,y_train)

r2_score(y_test,r_dt.predict(x_test)) # 0.6578111331866805

print(r_dt.predict(ceo))
print(r_dt.predict(mudur))
"""
#----------------------------------------------------------

#RF

from sklearn.ensemble import RandomForestRegressor

r_rf=RandomForestRegressor(n_estimators=20,random_state=0)
r_rf.fit(x_train,y_train)

r2_score(y_test,r_rf.predict(x_test)) # 0.7252519981047478

print(r_rf.predict(ceo))
print(r_rf.predict(mudur))