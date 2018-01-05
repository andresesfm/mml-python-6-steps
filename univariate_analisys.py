import numpy as np
import pandas as pd
from sklearn import datasets
import scipy as sp
import matplotlib.pyplot as plt

iris = datasets.load_iris()
#Convert to dataframe
iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                    columns=iris['feature_names'] + ['species'])

#replace values with class labels
iris.species  = np.where(iris.species == 0.0, 'setosa',
 np.where(iris.species == 1.0, 'versicolor','virginica'))

#remove spaces from column names
iris.columns = iris.columns.str.replace(' ','')
print(iris.describe())
print(iris['species'].value_counts())
#? plt.figsize(15,8)
iris.hist()
plt.suptitle("Histogram", fontsize=16)
plt.show()

iris.boxplot()
plt.title("Bar Plot", fontsize=16)
plt.show()

####Multivariate Analisys

#Mean for each column by species
print(iris.groupby(by="species").mean())
iris.groupby(by="species").mean().plot(kind="bar")

