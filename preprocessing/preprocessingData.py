# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 11:43:10 2022

@author: tolga
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fullData=pd.read_csv("C:/Software&Design/MachineLearning/News/lesson1/veriler.csv")
missingData= pd.read_csv("C:/Software&Design/MachineLearning/News/lesson1/eksikveriler.csv")
integerData=missingData.iloc[:,1:4]

# nan değerleri mean ile değiştirme

from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan, strategy="mean")
missingData[["yas"]]=imputer.fit_transform(integerData[["yas"]])

# # kategorik kolonu numeric kolona çevirme

from sklearn import preprocessing

# labelencoder ile categoric datayı unique sayılara çevirme
le = preprocessing.LabelEncoder() # pandas.Series ve np.array ile çalışabilirsin

missingData["ulke"]=le.fit_transform(missingData["ulke"]) # le 1,2,3,0 gibi kategorik verileri numerik değerlere çevirir

# onehotencoder ile categoric datayı sayı 1,0lı data colomnlarına çevirme

ohe=preprocessing.OneHotEncoder() # yalnızca np.array ile çalışabilirsin

country = missingData.iloc[:,0:1].values
newCountries=ohe.fit_transform(country).toarray()

missingData=missingData.iloc[:,1:5]

countriesColumn=pd.DataFrame(data=newCountries,index=range(len(missingData)),columns=["tr","us","fr"]) # pd.Seriesa çeviriyoruz.

editedData=pd.concat([countriesColumn,missingData],axis=1) # datayı sırasıyla x ekseninde birleştiriyor if x=0: data alt alta birleştirilir.


# datayı train ve test olarak ayırma

from sklearn.model_selection import train_test_split

x=editedData.iloc[:,0:6]
y=editedData.iloc[:,6]

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

