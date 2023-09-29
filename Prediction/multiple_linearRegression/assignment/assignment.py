import pandas as pd

data=pd.read_csv("C:/Software&Design/MachineLearning/News/prediction2/assignment/tenis.csv")

"""
ohe ile hava durumunu değiştir
le ile windy değiştir
le ile play değiştir
"""
from sklearn.preprocessing import OneHotEncoder

weather=data[["outlook"]]
ohe=OneHotEncoder()
newWeather=ohe.fit_transform(weather).toarray()
newWeather=pd.DataFrame(newWeather,index=range(len(data)),columns=["sunny","overcast","rainy"])

from sklearn.preprocessing import LabelEncoder
"""
Birden fazla kolona aynı işlemi uygulama

forLeData=data[["windy","play"]]
LeData=forLeData.apply(LabelEncoder().fit_transform())
"""
windy=data[["windy"]]
play=data[["play"]]
le=LabelEncoder()
newWindy=pd.DataFrame(le.fit_transform(windy),columns=["windy"])
newPlay=pd.DataFrame(le.fit_transform(play))
xData=data[["temperature","humidity"]]

x=pd.concat([newWeather,xData],axis=1)
x=pd.concat([x,newWindy],axis=1)
y=newPlay




import statsmodels.api as sm

x=x.iloc[:,[0,1,2]] #[0,1,2]

model=sm.OLS(y,x).fit()
model.summary()


from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)


from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)


