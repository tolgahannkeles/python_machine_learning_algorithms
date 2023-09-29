"""
prediction --> tüm zamanlarda tahmin
forecasting --> gelecekle ilgili tahmin
"""
# ----------------------------- PREPROCESSING --------------------------
# data yükleme
import pandas as pd
data=pd.read_csv("C:/Software&Design/MachineLearning/News/Prediction/prediction2/assignment/tenis.csv")

# data.info() # dosya hakkında brief verir


# Categoric --> Numeric

# Label Encoder
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit_transform()

# Onehot Encoder

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
ohe.fit_transform().toarray()




