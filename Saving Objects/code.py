import pandas as pd

url="https://bilkav.com/satislar.csv"
data=pd.read_csv(url)
data=data.values
x=data[:,:1]
y=data[:,1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)




path="C:/Users/tolga/Desktop/lr.kayit"

import pickle

savedObject=pickle.dump(lr,open(path,"wb"))

loadedObject=pickle.load(open(path,"rb"))