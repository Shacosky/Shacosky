import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from datetime import datetime
from sklearn.model_selection import train_test_split

# X, y = make_blobs(n_samples=1000,n_features=2, centers=2, cluster_std=1.5)
# dictdata =  dict(feature1=X[:,0], feature2=X[:,1], type=y)
# df = pd.DataFrame(dictdata)

# now = datetime.now()
# filename = 'perceptron_data_%d%d%d.csv' %(now.year, now.month, now.day)
# df.to_csv(filename, sep = ';', index = False, encoding = 'utf-8')

# plt.title("Random data with 'make_blobs'", fontsize='small')
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, alpha=0.5, edgecolor='c')
# plt.show()

path = '..\\Machine-Learning\\Perceptron\\Csv_data\\perceptron_data_2018310.csv'
df = pd.read_csv(path, sep=';')
print('_'*60 + 'COLUMNS')
print(df.columns.values)
print('_'*60 + 'INFO')
print (df.info())
print('_'*60 + 'DESCRIBE')
print (df.describe().transpose())
print('_'*60 + 'SHAPE')
print (df.shape)
print('_'*60 + 'COUNT VALUE CLASSES')
print (df.loc[:,'type'].value_counts())
print('_'*60 + 'NULL VALUES')
print (df.isnull().sum())



X, y = df.loc[:, ['feature1', 'feature2']].values, df.loc[:,['type']].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

class SimplePerceptron():

    def __init__(self, eta):
        """
        :param eta: tasa de aprendizaje
        """
        self.eta = eta

    def zeta(self, X):
        """
        Calcula el producto de las entradas por sus pesos
        :param X: datos de entrenamiento con las caracteristicas. Array
        """
        zeta = np.dot(1, self.weights[0]) + np.dot(X, self.weights[1:])
        return zeta

    def predict(self, X):
        """
        Calcula la salida de la neurona teniendo en cuenta la función de activación
        :param X: datos con los que predecir la salida de la neurona. Array
        :return: salida de la neurona
        """
        output = np.where(self.zeta(X) >= 0.0, 1, 0)
        return output

    def fit(self, X, y):
        #Ponemos a cero los pesos
        self.weights = [0] * (X.shape[1] + 1)
        self.errors = []
        self.iteraciones = 0
        while True:
            errors = 0
            for features, expected in zip(X,y):
                delta_weight = self.eta * (expected - self.predict(features))
                self.weights[1:] += delta_weight * features
                self.weights[0] += delta_weight * 1
                errors += int(delta_weight != 0.0)
            self.errors.append(errors)
            self.iteraciones += 1
            if errors == 0:
                break

#Creamos una instancia de la clase
sp = SimplePerceptron(eta=0.1)
#Entrenamos
sp.fit(X_train, y_train)

#Comprobamos la precisión del perceptron con los datos de test
print('_'*60 + "Prediccion para X_test")
prediction = sp.predict(X_test)
print (prediction)
print('_'*60 + "Esperado para X_test")
print (y_test.T[0])
print('_'*60 + "¿Coincide lo esperado y lo devuelto por el perceptron?")
print (np.array_equal(prediction, y_test.T[0]))
print('_'*60 + "PRECISION")
print(str(np.sum(prediction == y_test.T[0])/prediction.shape[0] * 100) + ' %')