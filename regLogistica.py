import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression

steam_Base = pd.read_csv("steam.csv")

for i in range(0,len(steam_Base)):
    if steam_Base.loc[i,'platforms'] == 'windows':
        steam_Base.loc[i,'NoWindows']=0
    else:
        steam_Base.loc[i,'NoWindows']=1
        
values=['Action','Indie','Strategy','RPG','Racing','Casual']
steam_Base=steam_Base[steam_Base.genres.isin(values)]
print(steam_Base)
encoder=OrdinalEncoder(categories=[values])
encoder.fit(steam_Base[['genres']])
steam_Base["genres-encoded"] = encoder.transform(steam_Base[["genres"]])
print(steam_Base["genres-encoded"])
print(steam_Base.groupby('platforms').size())
steam_Base=steam_Base.drop(['platforms'],1)
print(steam_Base)

X = np.array(steam_Base['genres-encoded']).reshape(-1,1)
y = np.array(steam_Base['NoWindows'])

regresion_logistica = LogisticRegression()
regresion_logistica.fit(X,y)

prediccion = regresion_logistica.predict(X)
print(prediccion)
print(X)

probabilidades_prediccion = regresion_logistica.predict_proba(X)
print(probabilidades_prediccion)
print(probabilidades_prediccion[:,1])
print(probabilidades_prediccion[:,0])

plt.plot(X, y, 'ro')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Inicial')
plt.show()

plt.plot(X, prediccion, 'ro')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Predicción')
plt.show()

plt.plot(probabilidades_prediccion[:,0],probabilidades_prediccion[:,1])
plt.xlabel('X')
plt.ylabel('Y')
plt.xscale('logit')
plt.title('Regresión')
plt.show()
