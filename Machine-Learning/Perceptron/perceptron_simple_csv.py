import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#factor aprendizaje = lr
#epochs = numero de veces que ajusta el error

class Perceptron(object):
    "Implementa una red perceptron"
    def __init__(self,input_size, factor_aprendizaje=1,numero_iteraciones=100):
        self.weight = np.zeros(input_size+1)
        self.numero_iteraciones = numero_iteraciones
        self.factor_aprendizaje = factor_aprendizaje
        self.error = []
        
    def activation_fn(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.weight.T.dot(x)
        a = self.activation_fn(z)
        return a
    
    def fit(self, X, d):
        for _ in range(self.numero_iteraciones):
            for i in range(d.shape[0]):
                y = self.predict(X[i])
                e = d[i] - y
                self.weight = self.weight + self.factor_aprendizaje * e * np.insert(X[i], 0, 1)
                

if __name__ == "__main__":
    df = pd.read_csv('..\\Machine-Learning\\Perceptron\\Csv_data\\letters2.csv')
    count_classes = pd.value_counts(df['Clase'], sort = True)
    count_classes.plot(kind = 'bar', rot = 0)
    df.head(n=5)
    X = df.iloc[0:100, [0, 1]].values
    d = df.iloc[0:100, 2].values
    d = np.where(d == 'o', 0, 1)
    perceptron = Perceptron(input_size=2)
    perceptron.fit(X, d)

print(df)
plt.xticks(range(2),('Clase0','Clase1'))
plt.title("Frecuencia por numero de observacion")
plt.xlabel("Clase")
plt.ylabel("Numero de Observaciones")
plt.show()

plt.plot(range(1, len(perceptron.error) + 1), perceptron.error, marker='o')
plt.xlabel('Epochs')
plt.ylabel('errores')
plt.show()
