import pandas as pd
import numpy as np


data=pd.read_csv("C:/Stupid Humanity/Steps of AI Advanture/MachineLearning/Reinforced Learning/UCB/Ads_CTR_Optimisation.csv")

#UCB
N = len(data) # row sayısı
d = len(data.columns)  # column sayısı
#Ni(n)
tiklamalar = [0] * d #her kolondaki tıklama sayısı
#Ri(n)
oduller = [0] * d #her column için doğru tıklanma sayısı

# Sırasıyla tahmin edilen kolonlar
secilenler=[]

"""
Ni(n) # her rowun her columnı için tıklanma sayısı
Ri(n) # her columnın doğru tıklanma sayısı
ort=Ri(n)/Ni(n)
UCB=np.sqrt(3/2* np.log(column))/Ni(n)
her rowdaki kolonlar için bunu hesapla en büyük değere sahip olan rowu tahmin etmiş ol
"""


toplam = 0 #doğru tıklama sayısı

for n in range(0,N): #n= row
    ad=0
    max_ucb=0
    # max ucblı column seçme

    for i in range(0,d):
        # her columnın ucbsini hesapla
        if tiklamalar[i]>0:
            ortalama = oduller[i] / tiklamalar[i]
            delta = np.sqrt(3/2* np.log(n)/tiklamalar[i])
            ucb = ortalama + delta
        
        else:
            ucb=N*45613
        #max ucb yenileme ve kolonu ekleme
        if max_ucb<ucb:
            max_ucb=ucb
            ad=i

    #seçilen kolonu ödüllendirme
    odul=1
    if odul==data.values[n,ad]:
        oduller[ad]+=odul
    
    secilenler.append(ad)

    #tahmin sonucunu doğru sayılarını yansıtma
    toplam+=data.values[n,ad]
    tiklamalar[ad]+=1


    

print(f"Toplam Odul: {toplam}")