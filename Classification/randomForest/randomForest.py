import pandas as pd

data=pd.read_csv("C:/Software&Design/MachineLearning/News/Classification/classification6/veriler.csv")

x=data[["boy","kilo","yas"]]
y=data[["cinsiyet"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=0)


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(criterion="entropy",n_estimators=10)
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)