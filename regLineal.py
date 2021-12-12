import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import OrdinalEncoder



steam_csv = "steam.csv"
steam_df = pd.read_csv(steam_csv)
steam_df = steam_df.head(100)
enc = OrdinalEncoder()
owns = enc.fit_transform(steam_df[['owners']])
owns = owns.flatten()

print(owns)

X_train, X_test, y_train, y_test = train_test_split(
                                        steam_df.loc[:,['price']],
                                        owns,
                                        random_state = 123,test_size=0.3)


#Instanciamos el modelo
lm = LinearRegression(fit_intercept=True, normalize=True)
#Entrenamos
lm.fit(X_train, y_train)
#Predecimos en train y test
predictions_train = lm.predict(X_train)
predictions_test = lm.predict(X_test)
#Realizamos métricas para comprobar lo buenos que es nuestro modelo
print("error absoluto medio de entrenamiento:",mean_absolute_error(predictions_train, y_train))
print("raiz del error absoluto medio de entrenamiento:",
      np.sqrt(mean_squared_error(predictions_train, y_train)))
print("error absoluto medio de prueba:",mean_absolute_error(predictions_test, y_test))
print("raiz del error absoluto medio de prueba:",
      np.sqrt(mean_squared_error(predictions_test, y_test)))
print("R2 de entrenamiento:",r2_score(predictions_train, y_train))
print("R2 de prueba:",r2_score(predictions_test, y_test))
print("promedio minimo:",np.min(owns))
print("promedio maximo;",np.max(owns)) 

print(pd.DataFrame(lm.coef_, X_train.columns, columns=['Coefficient']))

preds = pd.DataFrame({'real_values':y_test, 'predictions':predictions_test})

preds.plot(kind='bar',figsize=(18,8))
plt.grid(linewidth='2')
plt.grid(linewidth='2')
plt.grid(None)
plt.show()

print(X_test.shape)
print(X_train.shape)
print(y_test.shape)
print(y_train.shape)
print(predictions_test.shape)
print(predictions_train.shape)

plt.plot(X_test,y_test,'ro')
plt.plot(X_test,predictions_test)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regresión')
plt.show()