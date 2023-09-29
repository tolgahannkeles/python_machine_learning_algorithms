"""
Decision Treede datadadaki Y'den başka bir sonuç çıkmaz.
"""
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Software&Design/MachineLearning/News/prediction3/maaslar.csv")

X=data[["Egitim Seviyesi"]]
Y=data[["maas"]]

from sklearn.tree import DecisionTreeRegressor

r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)


plt.scatter(X,Y,color="red")
plt.plot(X,r_dt.predict(X),color="green")
plt.show()

import sklearn.preprocessing
