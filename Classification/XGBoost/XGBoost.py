"""
Extreme gradient boosting
"""

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

from xgboost import XGBClassifier
xcb=XGBClassifier()
xcb.fit(x_train,y_train)
y_pred=xcb.predict(x_test)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)
