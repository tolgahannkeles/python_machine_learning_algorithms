from tkinter.tix import Select
# istatistiksel olarak öznitelik seçer

from sklearn.feature_selection import chi2, SelectKBest
import pandas as pd


data=pd.read_csv("C:/Stupid Humanity/Steps of AI Advanture/MachineLearning/Dimension Reduction/wine.csv")



x=data.iloc[:,:-1]
y=data.iloc[:,-1]

x_chi2= SelectKBest(chi2, k=2).fit_transform(x,y)
print(x_chi2.shape)
print(x.shape)