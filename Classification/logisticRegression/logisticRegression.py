# Logistic Regression
import pandas as pd

data=pd.read_csv("C:/Software&Design/MachineLearning/News/Classification/classification1/veriler.csv")

x=data[["boy","kilo","yas"]]
y=data[["cinsiyet"]]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)



from sklearn.linear_model import LogisticRegression

logr= LogisticRegression()
logr.fit(x_train,y_train)

y_pred=logr.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_true=y_test,y_pred=y_pred) # doğru ve yanlış tahmin edilen verilerin sayılarını tablo biçiminde dönüdürür