import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

#Esta función nos ayuda a organizar las reglas en forma de .csv cuando son generadas
def inspect(results):
    rh          = [tuple(result[2][0][0]) for result in results]
    lh          = [tuple(result[2][0][1]) for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(rh, lh, supports, confidences, lifts))

dataframe = pd.read_csv('steam.csv')
#filtramos las primeras 500 filas de los datos
dataframe=dataframe.head(499)
records = []
print(dataframe)
#filtramos las columnas para solo tener developer publisher y owners
dataframe=dataframe.drop(dataframe.columns[[0,1,2,3,6,7,8,9,10,11,12,13,14,15,17]],axis='columns')
print(dataframe)
#creamos un nuevo .csv para despues leerlo
dataframe.to_csv("steam2.csv", index=False)
dataframe = pd.read_csv('steam2.csv', header=None)
print(dataframe)
#vaciamos el .csv en recors, para luego generar las reglas
for i in range(0, 500):
    records.append([str(dataframe.values[i,j]) for j in range(0, 2)])
association_rules = apriori(records, min_support=0.0045, min_confidence=0.4, min_lift=3, min_length=2)
association_results = list(association_rules)
resultDataFrame=pd.DataFrame(inspect(association_results), columns=['antecedente','consecuente','soporte','confiansa','lift'])
print(resultDataFrame)
resultDataFrame.to_csv("reglas.csv", index=False)
