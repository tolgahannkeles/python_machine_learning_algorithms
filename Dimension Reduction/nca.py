import pandas as pd

from sklearn.neighbors import NeighborhoodComponentsAnalysis

nca=NeighborhoodComponentsAnalysis(n_components=2,random_state=42)
nca.fit(x_scaled,y)
x_nda=nca.transform(x_scaled)
x_nda=pd.Dataframe(x_nda, columns=["p1","p2"])