"""
prediction --> tüm zamanlarda tahmin
forecasting --> gelecekle ilgili tahmin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Software&Design/MachineLearning/News/Prediction/prediction1/data.csv")

from sklearn.model_selection import train_test_split

x=data[["Aylar"]]
y=data[["Satislar"]]

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

# X_train=sc.fit_transform(x_train)
# X_test=sc.fit_transform(x_test)
# Y_train=sc.fit_transform(y_train)
# Y_test=sc.fit_transform(y_test)


# Linear Regression 
from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x_train,y_train)

lr.intercept_ # y eksenini kesetiği nokta ax+b nin bsi
lr.coef_ # doğrunun eğimi ax+b nin asi


predict=lr.predict(x_test)

x_train=x_train.sort_index() # ilkini rando yaptığımız için 
y_train=y_train.sort_index()

# Visualizing
plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))
plt.title("Prediction of Sales Revenue")
plt.xlabel("Months")
plt.ylabel("Revenue")
plt.legend()
plt.show()

