import pandas as pd
import numpy as np

data=pd.read_csv("C:/Software&Design/MachineLearning/News/Deep Learning/Churn_Modelling.csv")
data=data.iloc[:,3:].values

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data[:,2]=le.fit_transform(data[:,2]) #female 0


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],
                        remainder="passthrough"
                        )
data = ohe.fit_transform(data)


x=data[:,:-1]
y=data[:,-1:]
y = np.asarray(y).astype(np.float64)
#------------
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)


import keras
from keras.models import Sequential
from keras.layers import Dense

#hidden layerlarda linear functionlar kullan output layerlarda sigmoid functionlar kullan

classifier=Sequential()
classifier.add(Dense(6,activation="relu",input_dim=12)) #1.hidden layer 
classifier.add(Dense(6,activation="relu")) #2.hidden layer 
classifier.add(Dense(1,activation="sigmoid")) #output layer

classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

classifier.fit(X_train,y_train,epochs=50)
y_pred=classifier.predict(X_test)

y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred=y_pred,y_true=y_test)
print(cm)