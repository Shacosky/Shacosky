import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#w es el peso de cada entrada
#X arreglo de inputs
#epochs numero de veces que ajustamos los pesos
#theta es el umbral  = -(z*e) = w0
#fac_ap: Factor de aprendizaje
#z es el valor que da el error perceptron
#e = (y-z)



lista_errores = []

def activation_fn(z):
    return 1 if z >= 0 else 0

def entrenar(theta, fac_ap, w1, w2,w3,w4, epochs,X, d, n_muestras):
    errores = True
    while errores:
        errores = False        
        for i in range(n_muestras):
            z = ((X[i][0] * w1) + (X[i][1] * w2) + (X[i][2] * w3) + (X[i][3] * w4)) - theta
            z = activation_fn(z)
      
            if z != d[i]:
                errores = True
                #caclular error
                error = (d[i] - z) * theta
                lista_errores.append(error)
                #ajustar theta
                theta = theta + (-(fac_ap * error))
                
                w1 = w1 + (X[i][0] * error * fac_ap)
                w2 = w2 + (X[i][1] * error * fac_ap)
                w3 = w3 + (X[i][2] * error * fac_ap)
                w4 = w4 + (X[i][3] * error * fac_ap)
                epochs +=1
            else:
                lista_errores.append(0)        

    return w1,w2,w3,w4,epochs,theta,lista_errores

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

theta = 0.2
fac_ap = 1
w1 = 0
w2 = 0
w3 = 0
w4 = 0
epochs = 0
d = np.array([0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]) #OR
n_muestras = len(d)
w1,w2,w3,w4,epochs,theta,lista_errores = entrenar(theta, fac_ap, w1, w2,w3,w4, epochs,X, d, n_muestras)
print(str(w1) + " -> peso w1")
print(str(w2) + " -> peso w2")
print(str(w3) + " -> peso w3")
print(str(w4) + " -> peso w4")
print(str(theta) + " -> peso theta ")
print(str(epochs) + " -> epochs")
print(lista_errores)

plt.plot(range(1, len(lista_errores) + 1), lista_errores, marker='o')
plt.xlabel('Epochs')
plt.ylabel('errores')
plt.show()