# Tratamiento de datos
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
# Gráficos
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Preprocesado y modelado
# ------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import graphviz as viz

# Configuración warnings
# ------------------------------------------------------------------------------
import warnings
warnings.filterwarnings('once')


steam_csv = "./archive/steam.csv"
steam_df = pd.read_csv(steam_csv)
steam_df = steam_df

# División de los datos en train y test
# ------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
                                        steam_df.loc[:,['positive_ratings',
                                                        'negative_ratings']],
                                        steam_df[['owners']],
                                        random_state = 123,test_size=0.4
                                    )

# Se identifica el nombre de las columnas numéricas 
numeric_cols = X_train.select_dtypes(include=['float64', 'int']).columns.to_list()


labels = np.concatenate([numeric_cols])


# Creación del modelo
# ------------------------------------------------------------------------------
modelo = DecisionTreeClassifier(
            max_depth         = 5,
            criterion         = 'gini',
            random_state      = 123
          )

# Entrenamiento del modelo
# ------------------------------------------------------------------------------
modelo.fit(X_train, y_train)


# Estructura del árbol creado
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6))

print(f"Profundidad del árbol: {modelo.get_depth()}")
print(f"Número de nodos terminales: {modelo.get_n_leaves()}")

plot = plot_tree(
            decision_tree = modelo,
            feature_names = labels.tolist(),
            class_names   = steam_df['owners'].values,
            filled        = True,
            impurity      = False,
            fontsize      = 6,
            ax            = ax,
       )

fig=export_graphviz(modelo,
                    out_file=None,
                    class_names=steam_df['owners'].values,
                    feature_names=labels.tolist(),
                    impurity=False,
                    filled=True)
grahp=viz.Source(fig, format="svg")

grahp.render("decision_tree_Chido")

# Error de test del modelo
#-------------------------------------------------------------------------------
predicciones = modelo.predict(X = X_test)

print("Matriz de confusión")
print("-------------------")
print(confusion_matrix(
    y_true    = y_test,
    y_pred    = predicciones
))

accuracy = accuracy_score(
            y_true    = y_test,
            y_pred    = predicciones,
            normalize = True
           )
print(f"El accuracy de test es: {100 * accuracy} %")

# Post pruning (const complexity pruning) por validación cruzada (solo para c4.5)
# ------------------------------------------------------------------------------
'''# Valores de ccp_alpha evaluados
param_grid = {'ccp_alpha':np.linspace(0, 5, 10)}

# Búsqueda por validación cruzada
grid = GridSearchCV(
        # El árbol se crece al máximo posible antes de aplicar el pruning
        estimator = DecisionTreeClassifier(
                            max_depth         = None,
                            criterion         = 'gini',
                            min_samples_split = 2,
                            min_samples_leaf  = 1,
                            random_state      = 123
                       ),
        param_grid = param_grid,
        scoring    = 'accuracy',
        cv         = 10,
        refit      = True,
        return_train_score = True
      )

grid.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(6, 3.84))
scores = pd.DataFrame(grid.cv_results_)
scores.plot(x='param_ccp_alpha', y='mean_train_score', yerr='std_train_score', ax=ax)
scores.plot(x='param_ccp_alpha', y='mean_test_score', yerr='std_test_score', ax=ax)
ax.set_title("Error de validacion cruzada vs hiperparámetro ccp_alpha");

# Mejor valor ccp_alpha encontrado
# ------------------------------------------------------------------------------
print("Mejor valor ccp_alpha encontrado", grid.best_params_)

# Estructura del árbol final
# ------------------------------------------------------------------------------
modelo_final = grid.best_estimator_
print(f"Profundidad del árbol: {modelo_final.get_depth()}")
print(f"Número de nodos terminales: {modelo_final.get_n_leaves()}")
# Error de test del modelo final
#-------------------------------------------------------------------------------
predicciones = modelo_final.predict(X = X_test)

accuracy = accuracy_score(
            y_true    = y_test,
            y_pred    = predicciones,
            normalize = True
           )
print(f"El accuracy de test postpoda es: {100 * accuracy} %")'''

print("Importancia de los predictores en el modelo")
print("-------------------------------------------")
importancia_predictores = pd.DataFrame(
                            {'predictor': labels.tolist(),
                             'importancia': modelo.feature_importances_}
                            )
print(importancia_predictores.sort_values('importancia', ascending=False))

# Predicción de probabilidades
#-------------------------------------------------------------------------------
predicciones = modelo.predict_proba(X = X_test)
# Probabilidad de clasificación según el modelo
# ------------------------------------------------------------------------------
df_predicciones = pd.DataFrame(data=predicciones, columns=steam_df['owners'].unique())
df_predicciones.to_csv('predicciones.csv',index=False)