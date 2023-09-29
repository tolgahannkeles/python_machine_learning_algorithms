"""
R2 hesaplaması:
    hata kareleri toplamı(HKT) = sum(y-y_pred)**2
    ortalama farkların toplamı(OFT)= sum(y-average(y_pred))**2

    R2=1-HKT/OFT

    R2 1e yakın olmalı
"""

"""
Adjusted R2 hesaplaması:

    n= eleman sayısı
    p= değişken sayısı
    Adjusted R2=1-(1-R2)*(n-1)/(n-p-1)

"""

from sklearn.metrics import r2_score
r2_score(y,y_pred) # asıl değer, predict değer

data.corr() # korelasyon çıkarır