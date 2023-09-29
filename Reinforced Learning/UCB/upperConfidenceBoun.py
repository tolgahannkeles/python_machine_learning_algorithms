import pandas as pd
import numpy as np


data=pd.read_csv("C:/Stupid Humanity/Steps of AI Advanture/MachineLearning/Reinforced Learning/UCB/Ads_CTR_Optimisation.csv")


"""
#Random selection ile tahmin yaptırma
N=len(df)
toplam=0

for n in range(0,N):
    data=df.iloc[n] # satırı alır
    ad=random.randrange(len(df.columns)) #column için random int üretir
    odul=1
    if odul==data[ad]: # eğer int doğruysa ödüle 1 ekler değilse bir şey yapmaz
        toplam+=odul
    else:
        toplam=toplam
    
    
"""
# -------------------- UCB ----------------
"""
https://prnt.sc/xjzGDLedu6DX
"""
import math
import matplotlib.pyplot as plt
#UCB
N = 10000 # 10.000 tıklama
d = 10  # toplam 10 ilan var
#Ri(n)
oduller = [0] * d #ilk basta butun ilanların odulu 0
#Ni(n)
tiklamalar = [0] * d #o ana kadarki tıklamalar
toplam = 0 # toplam odul
secilenler = []

for n in range(0,N):
    ad = 0 #seçilen ilan
    max_ucb = 0
    for i in range(0,d):
        if(tiklamalar[i] > 0):
            ortalama = oduller[i] / tiklamalar[i]
            delta = math.sqrt(3/2* math.log(n)/tiklamalar[i])
            ucb = ortalama + delta
        else:
            ucb = N*10
        if max_ucb < ucb: #max'tan büyük bir ucb çıktı
            max_ucb = ucb
            ad = i          
    secilenler.append(ad)
    tiklamalar[ad] = tiklamalar[ad]+ 1
    odul = data.values[n,ad] # verilerdeki n. satır = 1 ise odul 1
    oduller[ad] = oduller[ad]+ odul
    toplam = toplam + odul
print('Toplam Odul:')   
print(toplam)

# plt.hist(secilenler)
# plt.show()






