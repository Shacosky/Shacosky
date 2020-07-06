import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap


def tanh(x):
    return (1.0 - np.exp(-2*x))/(1.0 + np.exp(-2*x))

class Perceptron(object):
    "Implementa una red perceptron"
    def __init__(self,input_size, factor_aprendizaje= 1 ,numero_iteraciones=10):
        self.activity = tanh
        self.weight = np.zeros(input_size+1)
        self.numero_iteraciones = numero_iteraciones
        self.factor_aprendizaje = factor_aprendizaje
        self.error = []
        


    def activation_fn(self, x):
        # z = 1/(1 + np.exp(-x)) 
        return 1 if x >= 0 else 0
        # return z
    
    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.weight.T.dot(x)#Producto punto de los valores internos de X
        a = self.activation_fn(z)
        return a
    
    def fit(self, X, d):
        errores = True
        while errores:
            errores = False
            for i in range(self.numero_iteraciones):
                y = self.predict(X[i])
                if y != d[i]:
                    errores = True
                    e = (d[i] - y)
                    self.weight = self.weight + (self.factor_aprendizaje * e * np.insert(X[i], 0, 1))
                    self.error.append(e)
                else:
                    self.error.append(0)


    def predict_single_data(self, x):
        val = np.concatenate((np.ones(1).T, np.array(x)))
        for i in range(0,len(self.weight)):
            val = self.activity(np.dot(val,self.weight[i]))
            val = np.concatenate((np.ones(1).T,np.array(val)))
        return val[1]


if __name__ == '__main__':
    X = np.array([
        [0,0,0,0],
        [0,0,0,1],
        [0,0,1,0],
        [0,0,1,1],
        [0,1,0,0],
        [0,1,0,1],
        [0,1,1,0],
        [0,1,1,1],
        [1,0,0,0],
        [1,0,0,1],
        [1,0,1,0],
        [1,0,1,1],
        [1,1,0,0],
        [1,1,0,1],
        [1,1,1,0],
        [1,1,1,1],
    ])
    
d = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]) #AND
#d = np.array([0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]) #OR

perceptron = Perceptron(input_size=4)
perceptron.fit(X, d)



print(perceptron.error)
print(str(len(perceptron.error)) + "-> Numero de veces que se itero el peso")
print(str(perceptron.numero_iteraciones)+ "-> numero de iteraciones")

plt.plot(range(1, len(perceptron.error) + 1), perceptron.error, marker='o')
plt.xlabel('Epochs')
plt.ylabel('errores')
plt.show()

errores = []
for s in X:
    print(s,perceptron.predict_single_data(s))
    errores.append(perceptron.predict_single_data(s))

plt.figure(1)
plt.plot(errores)
plt.xlabel('Epochs')
plt.ylabel('errores')
plt.grid()
plt.show()




