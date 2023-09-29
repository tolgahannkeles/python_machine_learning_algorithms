import pandas as pd

data=pd.read_csv("C:/Software&Design/MachineLearning/News/prediction2/veriler.csv")

"""
H0 olmasını istediğimiz durum "null hypothesis, farksızlık hipotezi, sıfır hipotezi, boş hipotez"
H1 olmamasını istediğimiz durum "alternatif hipotez"
p-value genelde 0.05 alırız
p-value<0.05 H1 olma olasılığı artar 
p-value>0.05 H0 olma olasılığı artar 
"""

"""
# Değişken kullanımı

-Bütün değişkenleri almak
-Backward elimination
-Forward Selection
-Bidirectional elimination  
-Score Comparison

"""

from sklearn.preprocessing import OneHotEncoder

countries=data[["ulke"]]
ohe=OneHotEncoder()
editedCountries=ohe.fit_transform(countries).toarray()


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
gender=data[["cinsiyet"]]
editedGender=le.fit_transform(gender)


xData=data[["boy","kilo","yas"]]
print(type(editedGender))

editedCountries=pd.DataFrame(editedCountries,index= range(len(xData)),columns=["tr","us","us"])
editedGender=pd.DataFrame(editedGender,columns=["cinsiyet"])

data=pd.concat([editedCountries,xData,editedGender],axis=1)


from sklearn.model_selection import train_test_split
x=data.iloc[:,:-1]
y=data.iloc[:,-1]

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)


from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)


##------- backward elimation ---------
#step1
import statsmodels.api as sm

x=data.iloc[:,[0,1,2,3,4,5]]
model=sm.OLS(y,x).fit()
print(model.summary())

#step2
import statsmodels.api as sm
x=data.iloc[:,[0,1,3,4,5]]
model=sm.OLS(y,x).fit()
print(model.summary())

#step3
import statsmodels.api as sm
x=data.iloc[:,[0,1,3,4]]
model=sm.OLS(y,x).fit()
print(model.summary())


x_train1, x_test1,y_train1,y_test1 = train_test_split(x,y,test_size=0.33, random_state=0)
regressor2=LinearRegression()
regressor2.fit(x_train1,y_train1)

y_pred2=regressor2.predict(x_test1)
