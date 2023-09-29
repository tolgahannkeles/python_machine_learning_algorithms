"""
SVR da standartscaler kullanmak zorundasın!!!
svr yöntemler:
kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} 
"""


import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Software&Design/MachineLearning/News/prediction4/maaslar.csv")

x=data[["Egitim Seviyesi"]]
y=data[["maas"]]

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
X=sc1.fit_transform(x)
sc2=StandardScaler()
Y=sc2.fit_transform(y)

from sklearn.svm import SVR
svr=SVR(kernel="rbf")
svr.fit(X,Y)

plt.scatter(X,Y)
plt.plot(X,svr.predict(X))
plt.show()