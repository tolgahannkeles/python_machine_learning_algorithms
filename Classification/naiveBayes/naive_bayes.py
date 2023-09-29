"""
GaussianNB
    -continuos değerlerde kullanılır
MultinomialNB
    - multinominal değerlerde kullanılır
BernoulliNB
    -Bynominal datalarda kullanılır
"""



import pandas as pd

data=pd.read_csv("C:/Software&Design/MachineLearning/News/Classification/classification1/veriler.csv")

x=data[["boy","kilo","yas"]]
y=data[["cinsiyet"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=0)

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

gnb=GaussianNB()
gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_test)

