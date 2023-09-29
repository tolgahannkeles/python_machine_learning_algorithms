"""
Random forest datayı belirli parçalara bölerek o parçalara decision tree uygular.
classificationda çoğunluğun dediğini
decisionda sonuçların ortalamasını çevirir
"""

import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Software&Design/MachineLearning/News/prediction3/maaslar.csv")

x=data[["Egitim Seviyesi"]]
y=data[["maas"]]

from sklearn.ensemble import RandomForestRegressor
r_rf=RandomForestRegressor(n_estimators=20,random_state=0) # decision tree sayısını gir
r_rf.fit(x,y)

plt.scatter(x,y,color="red")
plt.plot(x,r_rf.predict(x),color="blue")
plt.show()
