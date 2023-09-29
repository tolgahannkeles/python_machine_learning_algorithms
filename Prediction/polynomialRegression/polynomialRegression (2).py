import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Software&Design/MachineLearning/News/prediction3/maaslar.csv")

x=data[["Egitim Seviyesi"]]
y=data[["maas"]]



from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_regr=PolynomialFeatures(degree=31)
x_poly=poly_regr.fit_transform(x)

lr=LinearRegression()
lr.fit(x_poly,y)

plt.scatter(x,y)
plt.plot(x,lr.predict(x_poly), color = 'blue')
plt.show()
#---------------------------------------------

