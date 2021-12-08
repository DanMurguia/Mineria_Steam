import numpy as np
from numpy import unique
from numpy import where

import pandas as pd

import scipy
from scipy.cluster.hierarchy import dendrogram,linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb

import sklearn
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm
from sklearn.preprocessing import scale

np.set_printoptions(precision=4,suppress=True)
rcParams["figure.figsize"] =20,10
sb.set_style("whitegrid")

iris = datasets.load_iris()
#scale the data
data = scale(iris.data)
target = pd.DataFrame(iris.target)
variable_names = iris.feature_names
print(variable_names)


z = linkage(data,"ward")

#generate dendrogram
dendrogram(z,truncate_mode= "lastp", p =12, leaf_rotation=45,leaf_font_size=15, show_contracted=True)
plt.title("Truncated Hierachial Clustering Dendrogram")
plt.xlabel("Cluster Size")
plt.ylabel("Distance")
#divide the cluster
plt.axhline(y=15)
plt.axhline(5)
plt.axhline(10)
plt.show()

#based on the dendrogram we have two clusetes 
k =3 
#build the model
HClustering = AgglomerativeClustering(n_clusters=k , affinity="euclidean",linkage="ward")
#fit the model on the dataset
fit = HClustering.fit(data)
#accuracy of the model
sm.accuracy_score(target,HClustering.labels_)
clusters = unique(fit)
# creación del grafico de dispersión
for cluster in clusters:
	# obtenemos los indices de las filas 
	row_ix = where(fit == cluster)
	# creación de las dispersiones de las muestras 
	plt.scatter(data[row_ix, 0], data[row_ix, 1])
# muestra la grafica
plt.show()

