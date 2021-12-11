import matplotlib.pyplot as plt
import numpy as np

#preparamos los datos
X = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5]).reshape(-1,1)
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
#importamos la clase LogisticRegresion de scikit-learn
from sklearn.linear_model import LogisticRegression

#Creamos una instancia de la Regresión Logística
regresion_logistica = LogisticRegression()

#Entrena la regresión logística con los datos de entrenamiento
regresion_logistica.fit(X,y)
print(regresion_logistica)

#Usa el modelo entrenado para obtener las predicciones con datos nuevos
prediccion = regresion_logistica.predict(X)
print(prediccion)
#obtenemos las probabilidades de la predicción
probabilidades_prediccion = regresion_logistica.predict_proba(X)
print(probabilidades_prediccion)

#la primera columna es la probabilidad de suspender
#la segunda columna es la probabilidad de aprobar

#Mostramos ambas columnas por separado
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